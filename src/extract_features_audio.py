import torch
import h5py
import librosa
import pandas as pd
from tqdm import tqdm 
from transformers import logging
from transformers import AutoFeatureExtractor, ClapAudioModel

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = "laion/clap-htsat-fused"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name) 
model = ClapAudioModel.from_pretrained(encoder_name).to(device)

def load_data(data_path):

    data = {'train': [], 'val': [], 'test':[]}

    train = pd.read_csv(data_path + "train.csv")
    dev = pd.read_csv(data_path + "dev.csv")
    test = pd.read_csv(data_path + "test.csv")

    for i,row in train.iterrows():
        data['train'].append({'audio_id': row['file_name'], 'file_name': data_path + "train" + "/" + row['file_name']})

    for i,row in dev.iterrows():
        data['val'].append({'audio_id': row['file_name'], 'file_name': data_path + "dev" + "/" + row['file_name']})

    for i,row in test.iterrows():
        data['test'].append({'audio_id': row['file_name'], 'file_name': data_path + "test" + "/" + row['file_name']})

    return data

    

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')

    audio_id_all = list(df['audio_id'])
    file_name_all = list(df['file_name'])

    all_unique_file_ids = []

    for idx in tqdm(range(0, len(df), bs)):
        audio_ids = audio_id_all[idx:idx + bs]
        file_names = file_name_all[idx:idx + bs]
        audio_read = [librosa.resample(librosa.load(i, sr=16000)[0],orig_sr=16000,target_sr=48000) for i in file_names]
        audio_input = feature_extractor(audio_read, sampling_rate=48000, return_tensors="pt").to(device)
        with torch.no_grad():
            encodings = model(input_features=audio_input['input_features'],is_longer=audio_input['is_longer'],output_hidden_states=True)
            encodings = torch.flatten(encodings.last_hidden_state,2)
            encodings = encodings.permute(0,2,1).detach().cpu().numpy()

        for audio_id, encoding in zip(audio_ids, encodings):
            if str(audio_id) not in all_unique_file_ids:
                h5py_file.create_dataset(str(audio_id), (64, 768), data=encoding)
                all_unique_file_ids.append(str(audio_id))


data_dir = '/path_to_dir_with_train_dev_test_files'
features_dir = '/features'

data = load_data(data_dir)

encode_split(data, 'train')
encode_split(data, 'val')
encode_split(data, 'test')