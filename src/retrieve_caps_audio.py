import json
import torch
import faiss
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, ClapModel, AutoFeatureExtractor


def load_clotho_data(clotho_data_path):
    """We load in all images and only the train captions."""

    train = pd.read_csv(clotho_data_path + "train.csv")
    dev = pd.read_csv(clotho_data_path + "dev.csv")
    test = pd.read_csv(clotho_data_path + "test.csv")

    audios = []
    captions = []

    for i,row in train.iterrows():
        audios.append({'audio_id': row['file_name'], 'file_name': clotho_data_path + "train" + "/" + row['file_name']})
        captions.append({'audio_id': row['file_name'], 'caption': row['caption'], 'file_name': clotho_data_path + "train" + "/" + row['file_name']})

    for i,row in dev.iterrows():
        audios.append({'audio_id': row['file_name'], 'file_name': clotho_data_path + "dev" + "/" + row['file_name']})

    for i,row in test.iterrows():
        audios.append({'audio_id': row['file_name'], 'file_name': clotho_data_path + "test" + "/" + row['file_name']})
 
    return audios, captions

def filter_captions(data):

    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 512

    audio_ids = [d['audio_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np')['input_ids'].tolist()
    
    filtered_audio_ids, filtered_captions = [], []

    assert len(audio_ids) == len(caps) and len(caps) == len(encodings)
    for audio_id, cap, encoding in zip(audio_ids, caps, encodings):
        if len(encoding) <= 50:
            filtered_audio_ids.append(audio_id)
            filtered_captions.append(cap)

    return filtered_audio_ids, filtered_captions

def encode_captions(captions, model, tokenizer, device):

    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = tokenizer(captions[idx:idx+bs],padding=True, return_tensors="pt").to(device)
            encoded_captions.append(model.get_text_features(**input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

def encode_audio(audios, model, feature_extractor, device):

    audio_ids = [i['audio_id'] for i in audios]
    audio_paths = [i['file_name'] for i in audios]
    
    bs = 64	
    audio_features = []
    
    for idx in tqdm(range(0, len(audio_paths), bs)):
        audio_read = [librosa.resample(librosa.load(i, sr=16000)[0],orig_sr=16000,target_sr=48000) for i in audio_paths[idx:idx+bs]]
        audio_input = feature_extractor(audio_read, sampling_rate=48000, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_features.append(model.get_audio_features(**audio_input).detach().cpu().numpy())

    audio_features = np.concatenate(audio_features)

    return audio_ids, audio_features

def get_nns(captions, audios, k=15):
    xq = audios.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in zip(nns_list):
            if xb_image_ids[int(nn[0])] == image_id:
                continue
            good_nns.append(captions[int(nn[0])])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions
 
def main(): 

    clotho_data_path = '/path_to_data_folder_with_train_dev_test_csvs'
    
    print('Loading data')
    audios, captions = load_clotho_data(clotho_data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")    

    print('Filtering captions')    
    xb_audio_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, tokenizer, device)
    
    print('Encoding images')
    xq_audio_ids, encoded_audios = encode_audio(audios, clip_model, feature_extractor, device)
    
    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_audios)
    print(nns)
    retrieved_caps = filter_nns(nns, xb_audio_ids, captions, xq_audio_ids)

    print('Writing files')
    faiss.write_index(index, "index_caps")
    json.dump(captions, open('index_captions.json', 'w'))

    json.dump(retrieved_caps, open('retrieved_caps_clap.json', 'w'))

if __name__ == '__main__':
    main()




    

