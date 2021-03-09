import os
import sys
import time
import random
import pickle
import argparse
import numpy as np
import networkx as nx
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import graph as og
import walker as wk
import utils
import loader
import models.pair_embed_model as pair_model


def main():
    def str2bool(string):
        return string.lower() in ['yes', 'true', 't', 1]

    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--random_seed", type=float, default=42)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('-g', '--graph_file', type=str, default='')
    parser.add_argument('-e', '--node_embed_path', type=str, default='../data/NDFRT_DDA_LINE_embed.txt')
    parser.add_argument('--random_walk', type=str, default='random;node2vec')
    parser.add_argument('--weighted', type='bool', default=False)
    parser.add_argument('--walk_length', type=int, default=64)
    parser.add_argument('--number_walks', type=int, default=32)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--p', type=float, default=1.)
    parser.add_argument('--q', type=float, default=1.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pair_win_size', type=int, default=5)
    parser.add_argument('--lctx_win_size', type=int, default=3)
    parser.add_argument('--rctx_win_size', type=int, default=3)

    parser.add_argument('--num_neg_samples', type=int, default=5)
    parser.add_argument('--num_ctx_samples', type=int, default=5)

    parser.add_argument('--trivar_sample', type='bool', default=False)

    parser.add_argument('--node_embed_dim', type=int, default=128)
    parser.add_argument('--ctx_embed_dim', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs for training")
    parser.add_argument("--log_interval", type=int, default=20000, help='step interval for log')
    parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
    parser.add_argument("--early_stop_epochs", type=int, default=10)
    parser.add_argument("--metric", type=str, default='map', help='mrr or map')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--min_epochs', type=int, default=30, help='minimum number of epochs')
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--dev_log_interval', type=int, default=1000000)

    parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
    parser.add_argument("--save_dir", type=str, default='./saved_models',
                        help='save model in the best epoch or not')
    parser.add_argument("--save_interval", type=int, default=1, help='intervals for saving models')

    parser.add_argument('--data_workers', type=int, default=0)
    parser.add_argument('--walk_pos_train_path', type=str, default='')
    parser.add_argument('--walk_pos_dev_path', type=str, default='')
    parser.add_argument('--re_wallking', type='bool', default=False)

    args = parser.parse_args()
    print('args: ', args)
    print('\n' + '*'*10, 'Key parameters:', '*'*10)
    print('Use GPU? {0}'.format(torch.cuda.is_available()))
    print('Process id:', os.getpid())
    print('*'*35)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    og_graph = og.Graph()
    og_graph.read_edgelist(filename=args.graph_file, weighted=args.weighted)
    print('# nodes: ', len(og_graph.G.nodes))

    start = time.time()
    '''
    if not os.path.exists(args.walk_pos_train_path + '.txt') or args.re_wallking:
        print('Begin random walking ...')
        # generate random walks
        if args.random_walk == 'random':
            walker = wk.BasicWalker(og_graph, workers=args.workers)
        else:
            walker = wk.Walker(og_graph, p=args.p, q=args.q, workers=args.workers)
            walker.preprocess_transition_probs()
        sentences = walker.simulate_walks(num_walks=args.number_walks, walk_length=args.walk_length)
        print('Number of sequences: ', len(sentences))

        # generate pair contexts
        pair_ctxs, pair_count = utils.generate_pos_pairs(sentences,
                                                         args.pair_win_size,
                                                         args.lctx_win_size,
                                                         args.rctx_win_size)
        print('Total number of pairs: ', len(pair_ctxs))

        walk_train_file_name = '../data/{0}/{1}_walk_train_{2}_{3}_{4}_{5}_{6}'.format(args.dataset,
                                                                                       args.random_walk,
                                                                                       args.walk_length,
                                                                                       args.number_walks,
                                                                                       args.pair_win_size,
                                                                                       args.lctx_win_size,
                                                                                       args.rctx_win_size)

        with open(walk_train_file_name + '.txt', 'w') as f:
            for x in pair_ctxs:
                for y in pair_ctxs[x]:
                    f.write('{0} {1} {2}\n'.format(x[0], x[1], y))
        exit()
        # train_data = [[x[0], x[1], y] for x in pair_ctxs for y in pair_ctxs[x]]
        # np.random.shuffle(train_data)
        # print('Number of instances: ', len(train_data))
        # pickle.dump(train_data, open(walk_train_file_name + '.pkl', 'wb'), protocol=-1)
    else:
        print('Load random walking ...')
        # train_data = pickle.load(open(args.walk_pos_train_path, 'rb'))

        # train_data = open(args.walk_pos_train_path + '.txt').readlines()
        # train_data = [(int(x.strip().split()[0]),
        #                int(x.strip().split()[1]),
        #                int(x.strip().split()[2])) for x in train_data]
        # np.random.shuffle(train_data)
    '''
    # print('Data loaded! ', time.time() - start)
    # print(train_data[:10])

    # dev_data = train_data[-int(len(train_data) * 0.1)::]
    # train_data = train_data[: int(len(train_data) * 0.9)]
    # print(len(train_data), len(dev_data))

    node_to_id, id_to_node, node_mat = utils.node_id_mapping(args.node_embed_path)
    args.node_to_id = node_to_id
    args.id_to_node = id_to_node
    args.pre_train_nodes = node_mat
    args.node_embed_dim = node_mat.shape[1]
    args.node_vocab_size = len(node_to_id)
    print(len(args.node_to_id))
    print(len(og_graph.G.nodes))

    train_dataset = loader.PairDatasetIterable(args.walk_pos_train_path,
                                               args.node_to_id,
                                               args.num_neg_samples,
                                               args.num_ctx_samples,
                                               args.trivar_sample)

    # train_dataset = loader.PairDataset(train_data,
    #                                    args.node_to_id,
    #                                    args.num_neg_samples,
    #                                    args.num_ctx_samples,
    #                                    args.trivar_sample)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.data_workers,
                              worker_init_fn=lambda x: np.random.seed((torch.initial_seed() + x) % (2**32)))

    dev_dataset = loader.PairDatasetIterable(args.walk_pos_dev_path,
                                             args.node_to_id,
                                             args.num_neg_samples,
                                             args.num_ctx_samples,
                                             args.trivar_sample)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # for x in train_loader:
    #     print(x[0].shape)
    #     print(x[1].shape)
    #     print(x[2].shape)
    #     print(x[3].shape)
    #     if len(x) > 4:
    #         print(x[4].shape)
    #         print(x[5].shape)
    #     exit()

    # dev_dataset = loader.PairDataset(dev_data,
    #                                  args.node_to_id,
    #                                  args.num_neg_samples,
    #                                  args.num_ctx_samples,
    #                                  args.trivar_sample)
    # dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    model = pair_model.PairEmbedModel(args).to(args.device)
    print(model)
    print([name for name, p in model.named_parameters()])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    best_dev_loss, best_train_loss = np.inf, np.inf
    print('Begin trainning...')
    for epoch in range(args.num_epochs):
        print(datetime.now().strftime("%m/%d/%Y %X"))
        steps = 0
        train_loss = []
        start = time.time()
        for train_batch in train_loader:
            # print(train_batch)
            # exit()
            #
            model.train()
            optimizer.zero_grad()

            pair_embed, loss = model(train_batch)
            train_loss.append(loss.item() / train_batch[0].reshape(-1).shape[0])

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # print('----', time.time() - start, train_batch[0].shape, train_batch[0][0], train_batch[3][0])

            # evaluation
            steps += 1
            #
            # if steps % 11 == 0:
            #     print('---11---', time.time() - start)
            #     start = time.time()

            if steps % args.log_interval == 0:
                print("Epoch-{0}, steps-{1}: Train Loss - {2:.23}".format(epoch, steps, np.mean(train_loss)))

            if steps % args.dev_log_interval == 0:
                model.eval()
                dev_loss = []
                print(datetime.now().strftime("%m/%d/%Y %X"))
                for dev_batch in dev_loader:
                    dev_pair_embed, loss = model(dev_batch)
                    dev_loss.append(loss.item() / dev_batch[0].reshape(-1).shape[0])
                print('---Epoch-{0}, steps-{1}: Dev Loss - {2:.23}'.format(epoch, steps, np.mean(dev_loss)))

                if np.mean(train_loss) < best_train_loss:
                    best_train_loss = np.mean(train_loss)
                    if args.save_best:
                        utils.save(model, args.save_dir, 'best_train_snapshot', epoch, steps)
                train_loss = []

        utils.save(model, args.save_dir, 'epoch_train_snapshot', epoch, steps)

    return


if __name__ == '__main__':
    main()
