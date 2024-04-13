# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2.modeling_gpt2 import load_tf_weights_in_gpt2, GPT2LMHeadModel, GPT2MLP, GPT2Attention, GPT2Block, GPT2Model 

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

    
class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(
        self,
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor
        
class ThisGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        #print("this gpt2")

        #print("self.is_cross_attention = is_cross_attention", self.is_cross_attention, is_cross_attention)
        
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        
        if self.is_cross_attention:
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim), 
                                                                                  self.embed_dim) 
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            split_size = int(self.split_size / self.cross_attention_reduce_factor)
            head_dim = int(self.head_dim / self.cross_attention_reduce_factor)

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(split_size, dim=2)
            attention_mask = encoder_attention_mask

            query = self._split_heads(query, self.num_heads, head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, int(self.head_dim / self.cross_attention_reduce_factor))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class ThisGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        if config.add_cross_attention:
            self.crossattention = ThisGPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

class ThisGPT2Model(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     token_type_ids: Optional[torch.LongTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    #         input_shape = input_ids.size() + 64
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #         batch_size = input_ids.shape[0]
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #         batch_size = inputs_embeds.shape[0]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view(-1, input_shape[-1])
    #     if position_ids is not None:
    #         position_ids = position_ids.view(-1, input_shape[-1])

    #     if past_key_values is None:
    #         past_length = 0
    #         past_key_values = tuple([None] * len(self.h))
    #     else:
    #         past_length = past_key_values[0][0].size(-2)
    #     if position_ids is None:
    #         position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    #     # GPT2Attention mask.
    #     if attention_mask is not None:
    #         if batch_size <= 0:
    #             raise ValueError("batch_size has to be defined and > 0")
    #         attention_mask = attention_mask.view(batch_size, -1)
    #         # We create a 3D attention mask from a 2D tensor mask.
    #         # Sizes are [batch_size, 1, 1, to_seq_length]
    #         # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    #         # this attention mask is more simple than the triangular masking of causal attention
    #         # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    #         attention_mask = attention_mask[:, None, None, :]

    #         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    #         # masked positions, this operation will create a tensor which is 0.0 for
    #         # positions we want to attend and the dtype's smallest value for masked positions.
    #         # Since we are adding it to the raw scores before the softmax, this is
    #         # effectively the same as removing these entirely.
    #         attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    #         attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    #     # If a 2D or 3D attention mask is provided for the cross-attention
    #     # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    #     if self.config.add_cross_attention and encoder_hidden_states is not None:
    #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    #         encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    #         if encoder_attention_mask is None:
    #             encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
    #         encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    #     else:
    #         encoder_attention_mask = None

    #     # Prepare head mask if needed
    #     # 1.0 in head_mask indicate we keep the head
    #     # attention_probs has shape bsz x n_heads x N x N
    #     # head_mask has shape n_layer x batch x n_heads x N x N
    #     head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    #     if inputs_embeds is None:
    #         inputs_embeds = self.wte(input_ids)
    #         inputs_embeds = torch.cat((inputs_embeds, encoder_hidden_states), dim=1)
            
    #     position_embeds = self.wpe(position_ids)
    #     hidden_states = inputs_embeds + position_embeds

    #     if token_type_ids is not None:
    #         token_type_embeds = self.wte(token_type_ids)
    #         hidden_states = hidden_states + token_type_embeds

    #     hidden_states = self.drop(hidden_states)

    #     output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    #     if self.gradient_checkpointing and self.training:
    #         if use_cache:
    #             logger.warning_once(
    #                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #             )
    #             use_cache = False

    #     presents = () if use_cache else None
    #     all_self_attentions = () if output_attentions else None
    #     all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    #     all_hidden_states = () if output_hidden_states else None
    #     for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
    #         # Model parallel
    #         if self.model_parallel:
    #             torch.cuda.set_device(hidden_states.device)
    #             # Ensure layer_past is on same device as hidden_states (might not be correct)
    #             if layer_past is not None:
    #                 layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
    #             # Ensure that attention_mask is always on the same device as hidden_states
    #             if attention_mask is not None:
    #                 attention_mask = attention_mask.to(hidden_states.device)
    #             if isinstance(head_mask, torch.Tensor):
    #                 head_mask = head_mask.to(hidden_states.device)
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_states,)

    #         if self.gradient_checkpointing and self.training:

    #             def create_custom_forward(module):
    #                 def custom_forward(*inputs):
    #                     # None for past_key_value
    #                     return module(*inputs, use_cache, output_attentions)

    #                 return custom_forward

    #             outputs = torch.utils.checkpoint.checkpoint(
    #                 create_custom_forward(block),
    #                 hidden_states,
    #                 None,
    #                 attention_mask,
    #                 head_mask[i],
    #                 encoder_hidden_states,
    #                 encoder_attention_mask,
    #             )
    #         else:
    #             outputs = block(
    #                 hidden_states,
    #                 layer_past=layer_past,
    #                 attention_mask=attention_mask,
    #                 head_mask=head_mask[i],
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_attention_mask=encoder_attention_mask,
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #             )

    #         hidden_states = outputs[0]
    #         if use_cache is True:
    #             presents = presents + (outputs[1],)

    #         if output_attentions:
    #             all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
    #             if self.config.add_cross_attention:
    #                 all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    #         # Model Parallel: If it's the last layer for that device, put things on the next device
    #         if self.model_parallel:
    #             for k, v in self.device_map.items():
    #                 if i == v[-1] and "cuda:" + str(k) != self.last_device:
    #                     hidden_states = hidden_states.to("cuda:" + str(k + 1))

    #     hidden_states = self.ln_f(hidden_states)

    #     hidden_states = hidden_states.view(output_shape)
    #     # Add last hidden state
    #     if output_hidden_states:
    #         all_hidden_states = all_hidden_states + (hidden_states,)

    #     if not return_dict:
    #         return tuple(
    #             v
    #             for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
    #             if v is not None
    #         )

    #     return BaseModelOutputWithPastAndCrossAttentions(
    #         last_hidden_state=hidden_states,
    #         past_key_values=presents,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attentions,
    #         cross_attentions=all_cross_attentions,
    #     )


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)

