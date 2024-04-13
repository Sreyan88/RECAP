import pandas as pd
import numpy as np
import os
import argparse
os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM, AutoFeatureExtractor
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from src.audio_encoder_decoder import RECAP, RECAPConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel

from src.utils import *

PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 60

def get_model_and_auxiliaries(args):

    # register model types
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("recap", RECAPConfig)
    AutoModel.register(RECAPConfig, RECAP)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.encoder_name) 
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    model = RECAP.from_encoder_decoder_pretrained(args.encoder_name, args.decoder_name, cross_attention_reduce_factor=cross_attention_reduce_factor)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder   
    model.config.max_length = CAPTION_LENGTH   
    model.config.rag = not args.disable_rag
  

    for param in model.encoder.parameters():
        param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
                for name, param in model.decoder.named_parameters():
                    if 'encoder_attn' not in name:
                        param.requires_grad = False

    else:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))

    return model, tokenizer, feature_extractor

def get_data(tokenizer, max_length, args):

    data = load_data_for_training(args.annotations_path, args.captions_path)
    train_df = pd.DataFrame(data['train'])


    train_dataset = TrainDataset(
                        df=train_df,
                        features_path=os.path.join(args.features_dir,'train.hdf5'),
                        tokenizer=tokenizer,
                        rag=not args.disable_rag,
                        template_path=args.template_path,
                        k=args.k,
                        max_caption_length=max_length)

    return train_dataset

def main(args):

    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    output_dir = '{}_{}M_{}'.format(model_type, args.attention_size, args.decoder_name)

    output_dir = os.path.join(args.experiments_dir, output_dir)
    
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs, 
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate = args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs, 
        logging_strategy="epoch", 
        output_dir=output_dir, 
        overwrite_output_dir=True, 
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_dir", type=str, default="src/features/", help="Directory where cached input audio features are stored")
    parser.add_argument("--annotations_path", type=str, default="/home/sreyang/scratch.ramanid-prj/icassp/clotho/", help="Path to folder with train/dev/test CSVs")
    parser.add_argument("--experiments_dir", type=str, default="experiments/", help="Directory where trained models will be saved")

    parser.add_argument("--encoder_name", type=str, default="laion/clap-htsat-fused", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=7, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False, help="Whether to train the decoder in addition to the attention")

    parser.add_argument("--k", type=int, default=3, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="laion/clap-htsat-unfused", help="Visual encoder used for retreiving captions")
    parser.add_argument("--captions_path", type=str, default="src/retrieved_caps_clap.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")

    args = parser.parse_args()

    main(args)
