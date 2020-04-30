import torch
from torch.utils.data import Dataset
import pandas as pd


def padding(seq, maxlen):
    if len(seq) > maxlen:
        return seq[:maxlen]
    seq = [0]*(maxlen-len(seq)) + seq
    return seq


class SourceDataset(Dataset):
    def __init__(self, dataset, vocab, max_len):
        self.df = pd.read_csv(dataset, sep='\t', header=0)
        self._labels = self.df['label'].unique().tolist()
        self._labels.sort()
        self.vocab = vocab
        self.max_len = max_len

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, index):
        code, label = self.df[['code', 'label']].iloc[index]
        code_list = [self.vocab[char] if char in self.vocab else 0 for char in code]
        code_list = padding(code_list, self.max_len)
        data = np.array(code_list)
        label = label - 1
        return data, label

    def __len__(self):
        return len(self.df)