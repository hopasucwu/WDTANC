import numpy as np

from config import get_parameter
import soundfile as sf
import os
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import librosa

params = get_parameter()

def generate_train_data_list():
    datasets = os.listdir(params['train_data_root'])
    data_list = []
    for data_name in datasets:
        data_root = os.path.join(params['train_data_root'], data_name)
        wav_names = os.listdir(data_root)
        # print(data_root)
        for wav_name in wav_names[:10]:
            if wav_name == 'ch15.wav':
                continue
            data_list.append(os.path.join(data_root, wav_name))
    return data_list

class train_dataset(Dataset):
    def __init__(self):
        self.file_list = generate_train_data_list()
        self.Len = len(self.file_list)

    def __getitem__(self, index):
        wav_path = self.file_list[index]
        wf = sf.SoundFile(wav_path)
        data = np.array(wf.read(dtype = np.float32))
        data = data / np.max(np.abs(data))
        data = torch.from_numpy(data)
        return data

    def __len__(self):
        return self.Len
    
class ChunkSplitter(object):
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
    
    def split(self, data):
        input_data = data[:-params['n_fft']]
        target_data = data[params['n_fft']:]
        N = input_data.shape[-1]
        if N < self.chunk_size:
            return [input_data], [target_data]
        
        st = 0
        chunks_input, chunks_trg = [], []
        while st + self.chunk_size < N:
            chunk_data_in = input_data[st:st + self.chunk_size]
            chunk_data_trg = target_data[st:st + self.chunk_size]
            chunks_input.append(chunk_data_in)
            chunks_trg.append(chunk_data_trg)
            st += self.chunk_size
        return chunks_input, chunks_trg

class DataLoader(object):
    def __init__(self, dataset, chunk_size, batch_size=256, num_workers = 4):
        self.batch_size = batch_size
        self.splitter = ChunkSplitter(chunk_size)
        self.data_loader = torch.utils.data.DataLoader(dataset, 
                                                       batch_size = 1,
                                                       num_workers = num_workers,
                                                       collate_fn = self._collate)
    def _collate(self, batch):
        chunks_input, chunks_trg = [], []
        for data in batch:
            c_i, c_t = self.splitter.split(data)
            chunks_input += c_i
            chunks_trg += c_t
        return chunks_input, chunks_trg

    def _merge(self, chunk_list_input, chunk_list_trg):
        N = len(chunk_list_input)
        blist_input, blist_trg = [], []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            blist_input.append(chunk_list_input[s:s + self.batch_size])
            blist_trg.append(chunk_list_trg[s:s + self.batch_size])
        rn = N % self.batch_size
        return (blist_input, blist_trg, chunk_list_input[-rn:] if rn else [],
                 chunk_list_trg[-rn:] if rn else [])
    
    def __iter__(self):
        chunk_list_input, chunk_list_trg = [], []
        for c_i, c_t in self.data_loader:
            chunk_list_input += c_i
            chunk_list_trg += c_t
            batches_in, batches_trg, chunk_list_input, chunk_list_trg =\
                                     self._merge(chunk_list_input, chunk_list_trg)
            for i in range(len(batches_in)):
                batch_dict = dict()
                batch_dict['input'] = torch.cat(batches_in[i]).reshape(self.batch_size, -1)
                batch_dict['target'] = torch.cat(batches_trg[i]).reshape(self.batch_size, -1)
                yield batch_dict

def get_train_dataloader():
    dataset = train_dataset()
    params = get_parameter()
    return DataLoader(dataset, params['input_length'], batch_size=params['batch_size'], num_workers=1)
