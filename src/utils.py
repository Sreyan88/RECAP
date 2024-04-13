from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
import pandas as pd
import re

CAPTION_LENGTH = 60
SIMPLE_PREFIX = "This audio sounds like "

# def clean_caption(caption):

#     if isinstance(caption, list):
#         back = []
#         for cap in caption:
#             cap = cap.lower()         
#             cap = cap.replace(',', ' , ') 
#             cap = re.sub(' +', ' ', cap)
#             cap = cap.replace(' ,', ',')
#             cap = re.sub(r'[.]', '', cap)
#             cap = cap.strip()
#             cap += '.'
#             back.append(cap)
#         return back
#     else:
#         caption = caption.lower()         
#         caption = caption.replace(',', ' , ') 
#         caption = re.sub(' +', ' ', caption)
#         caption = caption.replace(' ,', ',')
#         caption = re.sub(r'[.]', '', caption)
#         caption = caption.strip()
#         caption += '.'
#         return caption

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k])
        # infix = '\n\n'.join(clean_caption(retrieved_caps[:k]))
        #  + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    # text_ids = tokenizer.encode(clean_caption(text), add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    if SIMPLE_PREFIX in pred:
        pred = pred.split(SIMPLE_PREFIX)[-1]
    else:
        pred = pred.split(SIMPLE_PREFIX.strip())[-1]
    pred = pred.strip()
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=60):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)
        # load precomputed features
        encoder_outputs = self.features[self.df['audio_id'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(clotho_data_path, caps_path=None):

    train = pd.read_csv(clotho_data_path + "train.csv")
    dev = pd.read_csv(clotho_data_path + "dev.csv")

    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))

    data = {'train': [], 'val': []}

    visited = []
    for i,row in train.iterrows():
        file_name = row['file_name']

        if file_name not in visited:
            visited.append(file_name)
            if caps_path is not None:
                caps = retrieved_caps[str(file_name)]
            else:
                caps = None
            samples = []
            all_sentences = list(train[train['file_name'] == file_name]['caption'])
            for sentence in all_sentences:
                samples.append({'file_name': clotho_data_path + "train" + "/" + file_name, 'audio_id': file_name, 'caps': caps, 'text': sentence})

            data['train'] += samples

    visited = []
    for i,row in dev.iterrows():
        file_name = row['file_name']

        if file_name not in visited:
            visited.append(file_name)
            if caps_path is not None:
                caps = retrieved_caps[str(file_name)]
            else:
                caps = None
            samples = []
            all_sentences = list(dev[dev['file_name'] == file_name]['caption'])
            for sentence in all_sentences:
                samples.append({'file_name': clotho_data_path + "dev" + "/" + file_name, 'audio_id': file_name, 'caps': caps, 'text': sentence})

            data['val'] += samples

    return data 

def load_data_for_inference(clotho_data_path, caps_path=None):

    test = pd.read_csv(clotho_data_path + "test.csv")
    dev = pd.read_csv(clotho_data_path + "dev.csv")

    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    visited = []
    for i,row in test.iterrows():
        file_name = row['file_name']
        if file_name not in visited:
            visited.append(file_name)
            if caps_path is not None:
                caps = retrieved_caps[str(file_name)]
            else:
                caps = None

            data['test'].append({'file_name': clotho_data_path + "test" + "/" + file_name, 'audio_id': file_name, 'caps': caps})

    visited = []
    for i,row in dev.iterrows():
        file_name = row['file_name']
        if file_name not in visited:
            visited.append(file_name)
            if caps_path is not None:
                caps = retrieved_caps[str(file_name)]
            else:
                caps = None
                
            data['val'].append({'file_name': clotho_data_path + "dev" + "/" + file_name, 'audio_id': file_name, 'caps': caps})

    return data     

