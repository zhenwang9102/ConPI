import os
import sys
import copy
import time
import random
import itertools
import numpy as np
from tqdm import tqdm
import networkx as nx
from datetime import datetime

import torch


def generate_neg_rela_pairs(pos_pairs, all_terms, seed=41, dev_ratio=0.15, test_ratio=0.15):
    start = time.time()
    num_samples = len(pos_pairs)
    neg_pairs = []
    pos_set = set(pos_pairs)
    np.random.seed(seed)
    while len(neg_pairs) < num_samples:
        t1, t2 = pos_pairs[np.random.randint(len(pos_pairs))]
        t1_sample = all_terms[np.random.randint(len(all_terms))]
        t2_sample = all_terms[np.random.randint(len(all_terms))]

        if t1 != t2_sample and (t1, t2_sample) not in pos_set and (t1, t2_sample) not in set(neg_pairs):
            neg_pairs.append((t1, t2_sample))

        if t1_sample != t2 and (t1_sample, t2) not in pos_set and (t1_sample, t2) not in set(neg_pairs):
            neg_pairs.append((t1_sample, t2))

    data = [(x[0], x[1], 1) for x in pos_pairs] + [(x[0], x[1], 0) for x in neg_pairs]
    np.random.shuffle(data)
    train_data = data[: int((1 - dev_ratio - test_ratio) * len(data))]
    dev_data = data[int((1 - dev_ratio - test_ratio) * len(data)): int((1 - test_ratio) * len(data))]
    test_data = data[int((1 - test_ratio) * len(data))::]

    print('Finish sampling negatives: ', time.time() - start)
    return train_data, dev_data, test_data


def generate_pos_pairs(sequences, pair_win_size, left_ctx_win_size, right_ctx_win_size):
    pair_ctx = {}
    pair_count = {}
    # for k in tqdm(range(len(sequences))):
    for k in range(len(sequences)):
        if k % 5000 == 0:
            print(k, datetime.now().strftime("%m/%d/%Y %X"))
        sequence = [int(x) for x in sequences[k]]
        for i in range(len(sequence)):
            js = [x for x in range(i + 1, i + pair_win_size) if x < len(sequence)]
            for j in js:
                pair = (sequence[i], sequence[j])
                pair_ctx.setdefault(pair, set())
                pair_count.setdefault(pair, 0)
                contexts = list(sequence[max(i - left_ctx_win_size, 0): i]) + \
                           list(sequence[i + 1: j]) + \
                           list(sequence[j + 1: min(j + right_ctx_win_size + 1, len(sequence) + 1)])
                pair_ctx[pair] = pair_ctx[pair].union(contexts)
                pair_count[pair] += 1

    return pair_ctx, pair_count


def node_id_mapping(embed_file, node_to_id=None):
    f = open(embed_file).readlines()
    print('Node Embeddings: ', f[0].strip())
    node_dict = {int(x.strip().split()[0]): np.array(x.strip().split()[1::], dtype=np.float32)
                 for x in f[1::]}
    # print(len(node_dict))
    if not node_to_id:
        node_embed_mat = np.zeros((len(node_dict) + 1, int(f[0].strip().split()[1])))
        node_to_id = {-1: 0}
        id_to_node = {0: -1}
        for idx, node_id in enumerate(node_dict.keys()):
            emb = node_dict[node_id].reshape(-1)
            node_embed_mat[idx + 1, :] = emb / np.linalg.norm(emb)
            node_to_id[node_id] = idx + 1
            id_to_node[idx + 1] = node_id

        return node_to_id, id_to_node, node_embed_mat
    else:
        node_embed_mat = np.zeros((len(node_to_id), int(f[0].strip().split()[1])))
        for node_id, idx in node_to_id.items():
            emb = node_dict[node_id].reshape(-1)
            node_embed_mat[idx, :] = emb / np.linalg.norm(emb)

        return node_embed_mat


def generate_all_neg_edges(G, train_G, dev_pos_edges, test_pos_edges, seed, sample_ratio=1, train_ratio=1, train_neg=None):
    start = time.time()
    all_nodes = list(G.nodes())
    # create a complete graph
    com_G = nx.Graph()
    com_G.add_nodes_from(all_nodes)
    com_G.add_edges_from(itertools.combinations(all_nodes, 2))
    # remove original edges
    com_G.remove_edges_from(G.edges())
    if train_neg:
        com_G.remove_edges_from(train_neg)

    train_pos_edges = train_G.edges
    num_train_neg = len(train_pos_edges) * sample_ratio * train_ratio
    num_dev_neg = len(dev_pos_edges) * sample_ratio
    num_test_neg = len(test_pos_edges) * sample_ratio
    total_neg_samples = num_train_neg + num_dev_neg + num_test_neg

    random.seed(seed)
    all_neg_edges = random.sample(com_G.edges, total_neg_samples)
    train_neg_edges = train_neg if train_neg else all_neg_edges[:num_train_neg]
    dev_neg_edges = all_neg_edges[num_train_neg: num_train_neg + num_dev_neg]
    test_neg_edges = all_neg_edges[-num_test_neg:]

    assert set(train_pos_edges).isdisjoint(set(dev_pos_edges))
    assert set(train_pos_edges).isdisjoint(set(test_pos_edges))
    assert set(dev_pos_edges).isdisjoint(set(test_pos_edges))
    assert set(train_neg_edges).isdisjoint(set(dev_neg_edges))
    assert set(train_neg_edges).isdisjoint(set(test_neg_edges))
    assert set(dev_neg_edges).isdisjoint(set(test_neg_edges))

    train_data = [(x[0], x[1], 1) for x in train_pos_edges] + [(x[0], x[1], 0) for x in train_neg_edges]
    dev_data = [(x[0], x[1], 1) for x in dev_pos_edges] + [(x[0], x[1], 0) for x in dev_neg_edges]
    test_data = [(x[0], x[1], 1) for x in test_pos_edges] + [(x[0], x[1], 0) for x in test_neg_edges]

    print('Time for sampling negatives: {0:.2f}s'.format(time.time() - start))
    return train_data, dev_data, test_data


def generate_neg_edges(original_graph, testing_edges_num, seed):
    L = list(original_graph.nodes())
    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save(model, save_dir, save_prefix, epoch, step=0):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    if step:
        save_path = '{0}_epoch_{1}_step_{2}.pt'.format(save_prefix, epoch, step)
    else:
        save_path = '{0}_epoch_{1}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)
