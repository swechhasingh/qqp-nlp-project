import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import (create_vocab,
                   tokenize_data,
                   handle_oov,
                   parse_n_store_glove,
                   sent_word2word_idx,
                   max_sent_len)


class QQPDataset(Dataset):
    """Quora Question Pairs dataset."""

    def __init__(self, data, split='train', vocab=None, word2idx=None, pre_process=None, device=None, debug=False):
        """
        Args:
                pre_process: ['remove_stopwords', 'stemming', 'lemmatization']
        """
        # data = data
        self.device = device
        self.pre_process = pre_process

        if debug:
            self.labels = torch.from_numpy(data['labels'][:100]).to(device)
            # tokenize data
            self.q1s = tokenize_data(data['q1s'][:100])
            self.q2s = tokenize_data(data['q2s'][:100])
        else:
            self.labels = torch.from_numpy(data['labels']).to(device)
            # tokenize data
            self.q1s = tokenize_data(data['q1s'])
            self.q2s = tokenize_data(data['q2s'])

        if pre_process:
            if 'remove_stopwords' in pre_process:
                self.q1s = remove_stopwords(self.q1s)
                self.q2s = remove_stopwords(self.q2s)
            elif 'stemming' in pre_process:
                self.q1s = stemming(self.q1s)
                self.q2s = stemming(self.q2s)
            elif 'lemmatization' in pre_process:
                self.q1s = lemmatization(self.q1s)
                self.q2s = lemmatization(self.q2s)

        # create vocab
        if split == 'train':
            # word2idx is dict of (word , word_id)
            self.vocab, self.word2idx = create_vocab(self.q1s, self.q2s, k=None)
            print("Vocabulary size: ", len(self.vocab))
        else:
            self.vocab, self.word2idx = vocab, word2idx
            self.q1s = handle_oov(self.q1s, self.vocab)
            self.q2s = handle_oov(self.q2s, self.vocab)

        # create mask
        max_seq_len, self.q1_lengths = max_sent_len(self.q1s)
        self.q1_lengths = torch.from_numpy(self.q1_lengths).to(self.device)
        self.q1_mask = torch.zeros((len(self.labels), max_seq_len))
        for i, l in enumerate(self.q1_lengths):
            self.q1_mask[i, :l] = 1.

        # padding q1s
        for i in range(len(self.q1s)):
            for j in range(max_seq_len - self.q1_lengths[i]):
                self.q1s[i].append('PAD')

        max_seq_len, self.q2_lengths = max_sent_len(self.q2s)
        self.q2_lengths = torch.from_numpy(self.q2_lengths).to(self.device)
        self.q2_mask = torch.zeros((len(self.labels), max_seq_len))
        for i, l in enumerate(self.q2_lengths):
            self.q2_mask[i, :l] = 1.

        # padding q2s
        for i in range(len(self.q2s)):
            for j in range(max_seq_len - self.q2_lengths[i]):
                self.q2s[i].append('PAD')

        self.q1s = sent_word2word_idx(self.q1s, self.word2idx)
        self.q2s = sent_word2word_idx(self.q2s, self.word2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'q1': torch.LongTensor(self.q1s[idx]).to(self.device),
                  'q2': torch.LongTensor(self.q2s[idx]).to(self.device),
                  'q1_len': self.q1_lengths[idx],
                  'q2_len': self.q2_lengths[idx],
                  'label': self.labels[idx]}
        # 'q1_mask': self.q1_mask[idx].to(self.device),
        # 'q2_mask': self.q2_mask[idx].to(self.device),
        return sample
