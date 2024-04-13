import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
import sys
import h5py
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from src.utils import load_data_for_inference, prep_strings, postprocess_preds

PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 60


def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""
    
    template = open(args.template_path).read().strip() + ' '

    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')

    out = []
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        audio_id = eval_df['audio_id'][idx]
        caps = eval_df['caps'][idx]
        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                 k=int(args.k), is_test=True)
        # load audio
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[audio_id][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        else:
            sys.exit(0)

        pred = tokenizer.decode(pred[0])

        pred = postprocess_preds(pred, tokenizer)
        out.append({"audio_id": audio_id, "caption": pred})

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.audio_encoder_decoder import RECAP, RECAPConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("recap", RECAPConfig)
    AutoModel.register(RECAPConfig, RECAP)

def main(args):

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.features_path is not None:
        feature_extractor = None
    else:
        feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)


    infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'test'

    data = load_data_for_inference(args.annotations_path, args.captions_path)

    eval_df = pd.DataFrame(data[split])
    args.outfile_name = '{}_preds.json'.format(split)

    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)

            infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_path", type=str, default="path_to/test.hdf5", help="H5 file with cached input audio features")
    parser.add_argument("--annotations_path", type=str, default="path_to_folder_with_train_test_dev_csvs/", help="Path to folder with train/dev/test CSV files")
        
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="laion/clap-htsat-fused", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="laion/clap-htsat-unfused", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="/home/sreyang/scratch.ramanid-prj/icassp/smallcap/src/retrieved_caps_clap_audiocaps.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    args = parser.parse_args()

    main(args)
   
