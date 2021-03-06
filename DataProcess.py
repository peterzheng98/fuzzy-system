import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np


def padding(seq, maxlen):
    for i, d in enumerate(seq):
        seq[i] = seq[i] + 1
    if len(seq) > maxlen:
        return seq[:maxlen]
    seq = [0]*(maxlen-len(seq)) + seq
    return seq


class SourceDataset(Dataset):
    def __init__(self, dataset, vocab, max_len):
        self.df = pd.read_csv(dataset)
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, index):
        sentence, label = self.df[['data', 'label']].iloc[index]
        sentence = json.loads(sentence)
        sentence_list = padding(sentence, self.max_len)
        data = np.array(sentence_list)
        label = 0 if label == 0 else 1
        return data, label

    def __len__(self):
        return len(self.df)


def linear_collate(samples):
    return torch.tensor(tuple([i[0] for i in samples])).long(), torch.tensor(tuple(i[1] for i in samples)).long()