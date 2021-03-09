import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


def ctx_pair_batching(data):
    t1_batch = []
    t2_batch = []
    t1_ctx_pair_batch = []
    t2_ctx_pair_batch = []
    labels = []
    for sample in data:
        t1_batch.append(sample['t1'])
        t2_batch.append(sample['t2'])
        t1_ctx_pair_batch.append(sample['t1_pairs'])
        t2_ctx_pair_batch.append(sample['t2_pairs'])
        labels.append(sample['y'])

    t1_batch = np.array(t1_batch)
    t2_batch = np.array(t2_batch)
    labels = np.array(labels)
    # padding and masking
    t1_pair_batch, t2_pair_batch, masks = padding_pairs(t1_ctx_pair_batch, t2_ctx_pair_batch)

    return t1_batch, t2_batch, t1_pair_batch, t2_pair_batch, masks, labels


def padding_pairs(t1_pair_batch, t2_pair_batch, min_len=0, max_len=None, padder=0):
    # [[1,2,3], [6]] [[3,4,5], [7]] -> [[1,2,3], [6,0,0]], [[3,4,5],[7,0,0]]
    if not max_len:
        max_len = max([len(x) for x in t1_pair_batch] + [min_len])
    new_t1_pair_batch = []
    new_t2_pair_batch = []
    masks = []
    for i in range(len(t1_pair_batch)):
        t1_pairs = t1_pair_batch[i]
        t2_pairs = t2_pair_batch[i]
        new_t1_pair_batch.append(t1_pairs + [padder] * (max_len - len(t1_pairs)))
        new_t2_pair_batch.append(t2_pairs + [padder] * (max_len - len(t2_pairs)))
        masks.append([1] * len(t1_pairs) + [0] * (max_len - len(t1_pairs)))

    return np.array(new_t1_pair_batch), np.array(new_t2_pair_batch), masks


def make_idx_ctx_pair_batch(dataset, neighbor_func, node_to_id, max_num_ctx=100):
    data = []
    list_data = []
    for sample in dataset:
        cur_dict = {}
        t1 = sample[0]
        t2 = sample[1]
        label = sample[-1]
        cur_dict['t1'] = node_to_id[int(t1)]
        cur_dict['t2'] = node_to_id[int(t2)]

        t1_ctx = [node_to_id[int(x)] for x in neighbor_func(t1)][:max_num_ctx]
        t2_ctx = [node_to_id[int(x)] for x in neighbor_func(t2)][:max_num_ctx]
        t1_pairs, t2_pairs = [], []
        for h in t1_ctx:
            for t in t2_ctx:
                t1_pairs.append(h)
                t2_pairs.append(t)

        cur_dict['t1_pairs'] = t1_pairs
        cur_dict['t2_pairs'] = t2_pairs
        cur_dict['y'] = label
        data.append(cur_dict)
        list_data.append(sample)

    return data, list_data


def make_idx_ctx_data(dataset, neighbor_func, node_to_id, max_num_ctx=100):
    data = []
    list_data = []
    for sample in dataset:
        cur_dict = {}
        t1 = sample[0]
        t2 = sample[1]
        label = sample[-1]
        t1_ctx = neighbor_func(t1)
        t2_ctx = neighbor_func(t2)

        cur_dict['t1'] = node_to_id[int(t1)]
        cur_dict['t2'] = node_to_id[int(t2)]

        cur_dict['t1_ctx'] = [node_to_id[int(x)] for x in t1_ctx][:max_num_ctx]
        cur_dict['t2_ctx'] = [node_to_id[int(x)] for x in t2_ctx][:max_num_ctx]

        cur_dict['y'] = label
        data.append(cur_dict)
        list_data.append(sample)

    return data, list_data


def ctx_batching(data):
    t1_batch = []
    t2_batch = []
    t1_ctx_batch = []
    t2_ctx_batch = []
    labels = []
    for sample in data:
        t1_batch.append(sample['t1'])
        t2_batch.append(sample['t2'])
        t1_ctx_batch.append(sample['t1_ctx'])
        t2_ctx_batch.append(sample['t2_ctx'])
        labels.append(sample['y'])

    t1_batch = np.array(t1_batch)
    t2_batch = np.array(t2_batch)
    labels = np.array(labels)
    return t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, labels


class PairDatasetIterable(IterableDataset):

    def __init__(self, filename, node_to_id, num_neg_samples, num_ctx_samples, trivar_sample=False):
        self.filename = filename
        self.node_to_id = node_to_id
        self.num_neg_samples = num_neg_samples
        self.num_ctx_samples = num_ctx_samples
        self.trivar_sample = trivar_sample

    def preprocess(self, text):
        t1, t2, ctx = text.strip().split()
        pos_t1 = np.array([self.node_to_id[int(t1)]])
        pos_t2 = np.array([self.node_to_id[int(t2)]])
        pos_ctx = np.array([self.node_to_id[int(ctx)]])
        sample_ctx = np.random.randint(1, len(self.node_to_id), self.num_ctx_samples)

        if self.trivar_sample:
            sample_t1 = np.random.randint(1, len(self.node_to_id), self.num_neg_samples)
            sample_t2 = np.random.randint(1, len(self.node_to_id), self.num_neg_samples)
            return pos_t1, pos_t2, pos_ctx, sample_ctx, sample_t1, sample_t2
        return pos_t1, pos_t2, pos_ctx, sample_ctx

    def line_mapper(self, line):
        text = self.preprocess(line)

        return text

    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr


class PairDataset(Dataset):
    def __init__(self, samples, node_to_id, num_neg_samples, num_ctx_samples, trivar_sample=False):
        self.data = {idx:x for idx, x in enumerate(samples)}
        self.node_to_id = node_to_id
        self.num_neg_samples = num_neg_samples
        self.num_ctx_samples = num_ctx_samples
        self.trivar_sample = trivar_sample

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        t1, t2, ctx = self.data[index]
        pos_t1 = np.array([self.node_to_id[int(t1)]])
        pos_t2 = np.array([self.node_to_id[int(t2)]])
        pos_ctx = np.array([self.node_to_id[int(ctx)]])
        sample_ctx = np.random.randint(1, len(self.node_to_id), self.num_ctx_samples)

        if self.trivar_sample:
            sample_t1 = np.random.randint(1, len(self.node_to_id), self.num_neg_samples)
            sample_t2 = np.random.randint(1, len(self.node_to_id), self.num_neg_samples)
            return pos_t1, pos_t2, pos_ctx, sample_ctx, sample_t1, sample_t2
        return pos_t1, pos_t2, pos_ctx, sample_ctx


class LinkPredDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, samples, node_to_id):
        'Initialization'
        self.head_dict = {}
        self.tail_dict = {}
        self.label_dict = {}
        for idx, sample in enumerate(samples):
            head = sample[0] if not isinstance(sample[0], str) else int(sample[0])
            tail = sample[1] if not isinstance(sample[1], str) else int(sample[1])
            self.head_dict[idx] = node_to_id[head]
            self.tail_dict[idx] = node_to_id[tail]
            self.label_dict[idx] = sample[-1]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.label_dict)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        head = self.head_dict[index]
        tail = self.tail_dict[index]
        label = self.label_dict[index]

        return head, tail, label
