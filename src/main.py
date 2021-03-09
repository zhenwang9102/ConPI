import os
import sys
import time
import copy
import pickle
import argparse
import numpy as np
import networkx as nx
from datetime import datetime
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import loader
import utils
import models.context_model as context_model
import models.context_pair_model as pair_model
import train_utils


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


parser = argparse.ArgumentParser(description='process user given parameters')
parser.register('type', 'bool', str2bool)
parser.add_argument("--random_seed", type=float, default=42)
parser.add_argument("--sample_seed", type=float, default=42)

parser.add_argument('--num_oov', type=int, default=2000)
parser.add_argument('--re_sample_test', type='bool', default=False)
parser.add_argument('--train_neg_num', type=int, default=50)
parser.add_argument('--test_neg_num', type=int, default=100)
parser.add_argument("--num_contexts", type=int, default=100, help="# contexts for interaction")
parser.add_argument('--max_contexts', type=int, default=1000, help='max contexts to look at')
parser.add_argument('--context_gamma', type=float, default=1)
# model parameters
parser.add_argument('--ngram_embed_dim', type=int, default=100)
parser.add_argument('--n_grams', type=str, default='2, 3, 4')
parser.add_argument("--word_embed_dim", type=int, default=100, help="embedding dimention for word")
parser.add_argument('--node_embed_dim', type=int, default=128)
parser.add_argument("--dropout", type=float, default=0, help="size of testing set")
parser.add_argument('--bi_out_dim', type=int, default=50, help='dim for the last bilinear layer for output')

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs for training")
parser.add_argument("--log_interval", type=int, default=2000, help='step interval for log')
parser.add_argument("--test_interval", type=int, default=1, help='epoch interval for testing')
parser.add_argument("--early_stop_epochs", type=int, default=10)
parser.add_argument("--metric", type=str, default='map', help='mrr or map')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--min_epochs', type=int, default=30, help='minimum number of epochs')
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')

parser.add_argument('--freeze_node', type='bool', default=True)

# path to external files
# parser.add_argument('--split_graph_path', type=str, default='../data/NDFRT_DDA_train_val_test.pkl')
# parser.add_argument('--graph_path', type=str, default='../data/NDFRT_DDA_graph.pkl')
parser.add_argument("--embed_filename", type=str, default='../data/embeddings/glove.6B.100d.txt')
parser.add_argument('--node_embed_path', type=str, default='../data/NDFRT_DDA_LINE_embed.txt')
parser.add_argument('--ngram_embed_path', type=str, default='../data/embeddings/charNgram.txt')
# parser.add_argument('--restore_para_file', type=str, default='./final_pretrain_cnn_model_parameters.pkl')
# parser.add_argument('--restore_model_path', type=str, required=True, default='')
parser.add_argument('--restore_idx_data', type=str, default='')
parser.add_argument("--logging", type='bool', default=False)
parser.add_argument("--log_name", type=str, default='empty.txt')
parser.add_argument('--restore_model_epoch', type=int, default=600)
parser.add_argument("--save_best", type='bool', default=True, help='save model in the best epoch or not')
parser.add_argument("--save_dir", type=str, default='./saved_models', help='save model in the best epoch or not')
parser.add_argument("--save_interval", type=int, default=5, help='intervals for saving models')

parser.add_argument('--log_reg_model', type='bool', default=False)
parser.add_argument('--max_num_ctx', type=int, default=100)
parser.add_argument('--ctx_model', type=str, default='static')

parser.add_argument('--re_sample_neg', type='bool', default=False)
parser.add_argument('--add_own_embed', type='bool', default=False)

parser.add_argument('--rela', type=str, default='clinically_associated_with')
parser.add_argument('--input_edgelist', type=str, default='')
parser.add_argument('--split_dataset_path', type=str, default='')

parser.add_argument('--graph_feature', type='bool', default=False)
parser.add_argument('--apply_sigmoid', type='bool', default=False)

parser.add_argument('--use_pair_pretraining', type='bool', default=False)
parser.add_argument('--restore_pair_embed', type=str, default='')
parser.add_argument('--ctx_embed_dim', type=int, default=128)
args = parser.parse_args()
print('args: ', args)

print('\n' + '*' * 10, 'Key parameters:', '*' * 10)
print('Use GPU? {0}'.format(torch.cuda.is_available()))
print('Process id:', os.getpid())
print('Relation: ', args.rela)
print('Embed path ', args.node_embed_path)
print('Using logistic regression? ', args.log_reg_model)
print('Using pre-trained pair embedding? ', args.use_pair_pretraining)
print('Interaction Model: ', args.ctx_model)
print('Max # contexts: ', args.max_num_ctx)
print('*' * 37)

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
args.cuda = torch.cuda.is_available()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.term_strings = pickle.load(open('../../SurfCon/global_mapping/term_string_mapping.pkl', 'rb'))
args.term_concept_dict = pickle.load(open('../../SurfCon/global_mapping/term_concept_mapping.pkl', 'rb'))
args.concept_term_dict = pickle.load(open('../../SurfCon/global_mapping/concept_term_mapping.pkl', 'rb'))


id_mapping = open('../BioNEV/data/Clin_Term_COOC/node_list.txt').readlines()
id_mapping = [x.strip().split() for x in id_mapping][1::]
# print(len(id_mapping))
args.nodeX_to_termid = {int(x[0]):int(x[1]) for x in id_mapping}
args.termid_to_nodeX = {int(x[1]):int(x[0]) for x in id_mapping}

# args.node_neighbor_dict = pickle.load(open('../data/node_embed_neighbor_dict.pkl', 'rb'))
# args.neighbors = pickle.load(open('../data/sub_neighbors_dict_ppmi_per' + args.per + '_' + args.days + '.pkl', 'rb'))
# args.all_iv_terms = list(args.neighbors.keys())
args.graph = nx.read_weighted_edgelist(args.input_edgelist)  # nodeX_id
# print(list(args.graph)[:10])
args.all_terms = set([int(x) for x in args.graph.nodes])  #nodeX
# args.all_terms = [args.nodeX_to_termid[int(x)] for x in args.graph.nodes]  # convert to term_id


all_triples = pickle.load(open('../data_rela/all_term_tuples_perBin_1.pkl', 'rb'))  # term id
# convert term_id to nodeX
pos_pairs = [(args.termid_to_nodeX[x[0]], args.termid_to_nodeX[x[1]]) for x in all_triples
             if x[-1] == args.rela and x[0] in args.termid_to_nodeX
             and x[1] in args.termid_to_nodeX]

# keep in term_id
# pos_pairs = [(x[0], x[1]) for x in all_triples
#              if x[-1] == args.rela and x[0] in args.termid_to_nodeX and x[1] in args.termid_to_nodeX]

print('Positive Data loaded!', len(pos_pairs))

print('Splitting data ...')
split_file = '{0}_{1}.pkl'.format(args.split_dataset_path, args.rela)
if not os.path.exists(split_file) or args.re_sample_neg:
    print('Re-sample negative pairs!')
    train_data, dev_data, test_data = utils.generate_neg_rela_pairs(pos_pairs,
                                                                    list(args.all_terms),
                                                                    args.random_seed)
    pickle.dump([train_data, dev_data, test_data], open(split_file, 'wb'), protocol=-1)
else:
    print('Loading negative pairs!')
    train_data, dev_data, test_data = pickle.load(open(split_file, 'rb'))

print(len(train_data), len(dev_data), len(test_data))

node_to_id, id_to_node, node_mat = utils.node_id_mapping(args.node_embed_path)
args.node_to_id = node_to_id
args.id_to_node = id_to_node
args.pre_train_nodes = node_mat
args.node_embed_dim = node_mat.shape[1]
args.node_vocab_size = len(node_to_id)

print('Begin digitalizing ...')
# prepare digital samples
# prepare the neighbor dict
neighbor_dict = lambda x: args.graph.adj[str(x)]

if not args.use_pair_pretraining:
    train_data, _ = loader.make_idx_ctx_data(train_data, neighbor_dict, args.node_to_id, args.max_num_ctx)
    dev_data, _ = loader.make_idx_ctx_data(dev_data, neighbor_dict, args.node_to_id, args.max_num_ctx)
    test_data, _ = loader.make_idx_ctx_data(test_data, neighbor_dict, args.node_to_id, args.max_num_ctx)
else:
    train_data, _ = loader.make_idx_ctx_pair_batch(train_data, neighbor_dict, args.node_to_id, args.max_num_ctx)
    dev_data, _ = loader.make_idx_ctx_pair_batch(dev_data, neighbor_dict, args.node_to_id, args.max_num_ctx)
    test_data, _ = loader.make_idx_ctx_pair_batch(test_data, neighbor_dict, args.node_to_id, args.max_num_ctx)

print(len(train_data), len(dev_data), len(test_data))
print(train_data[0])

# model selection
if not args.use_pair_pretraining:
    model = context_model.ContextInteractionModel(args).to(args.device)
else:
    model = pair_model.ContextPaireInteractionModel(args).to(args.device)

print(model)
print([name for name, p in model.named_parameters() if p.requires_grad==True])

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

last_epoch = 0
best_on_dev = 0.0
train_loss = []
train_logits = []
train_labels = []
best_res_dev = []
best_res_test = []
num_batches = len(train_data) // args.batch_size
print('Begin trainning...')
for epoch in range(args.num_epochs):
    # print(datetime.now().strftime("%m/%d/%Y %X"))
    model.train()
    steps = 0
    np.random.shuffle(train_data)
    for i in range(num_batches):
        train_batch = train_data[i * args.batch_size: (i + 1) * args.batch_size]
        if i == num_batches - 1:
            train_batch = train_data[i * args.batch_size::]

        if not args.use_pair_pretraining:
            t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, labels = loader.ctx_batching(train_batch)
            masks = None
        else:
            t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, masks, labels = loader.ctx_pair_batching(train_batch)
            t1_ctx_batch = torch.tensor(t1_ctx_batch, device=args.device)
            t2_ctx_batch = torch.tensor(t2_ctx_batch, device=args.device)
            masks = torch.tensor(masks, device=args.device)

        t1_batch = torch.tensor(t1_batch, device=args.device)
        t2_batch = torch.tensor(t2_batch, device=args.device)
        labels = torch.tensor(labels).type(torch.FloatTensor).to(args.device)

        optimizer.zero_grad()
        logits, _ = model(t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, masks=masks)
        loss = criterion(logits, labels)
        # print(type(labels), labels.shape)
        # print(type(logits), logits.shape)
        train_loss.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if args.cuda:
            logits = logits.to('cpu').detach().data.numpy()
            labels = labels.to('cpu').detach().data.numpy()
        else:
            logits = logits.detach().data.numpy()
            labels = labels.detach().data.numpy()

        # print(logits.shape)
        train_logits.append(logits)
        train_labels.append(labels)

        # evaluation
        steps += 1
        if steps % args.log_interval == 0:
            train_golds = np.concatenate(train_labels)
            train_logits = np.concatenate(train_logits)
            train_preds = np.where(utils.sigmoid(train_logits) >= 0.5, 1, 0)
            # train_preds = np.argmax(train_logits)
            # print(train_golds.shape, train_preds.shape)
            train_probs = utils.sigmoid(train_logits)
            train_acc = metrics.accuracy_score(train_golds, train_preds)
            train_f1 = metrics.f1_score(train_golds, train_preds)
            train_roc_auc = metrics.roc_auc_score(train_golds, train_probs)
            train_ap = metrics.average_precision_score(train_golds, train_probs)

            print("Epoch-{0}, steps-{1}: Train Loss - {2:.3}, Train F1 - {3:.4}".
                format(epoch, steps, np.mean(train_loss), train_f1*100))

            train_loss = []
            train_logits = []
            train_labels = []

    if epoch % args.test_interval == 0: d
        dev_results = train_utils.eval_link_context(dev_data, model, criterion, args)
        print("Epoch-{0}: Dev Acc {1:.4} F1 {2:.4} AUC {3:.4} AP {4:.4}".
              format(epoch, dev_results['acc']*100, dev_results['f1']*100,
                     dev_results['roc']*100, dev_results['ap']*100), end='')

        if dev_results['roc'] > best_on_dev:  # macro f1 score
            # print(datetime.now().strftime("%m/%d/%Y %X"))
            best_on_dev = dev_results['roc']
            best_res_dev = dev_results
            last_epoch = epoch

            test_results = train_utils.eval_link_context(test_data, model, criterion, args)
            print("\t-- Testing: Test Acc {0:.4} F1 {1:.4} AUC {2:.4} AP {3:.4}".
                format(test_results['acc']*100, test_results['f1']*100,
                       test_results['roc']*100, test_results['ap']*100))
            best_res_test = test_results

            if args.save_best:
                utils.save(model, args.save_dir, 'best', epoch)
            # print('\n')
        else:
            print('\r')
            if epoch - last_epoch > args.early_stop_epochs and epoch > args.min_epochs:
                print('Best Performance at epoch: {0}'.format(last_epoch))
                print('Best Dev: Acc {0:.4}\tF1 {1:.4}\tAUC {2:.4}\tAP {3:.4}'.format(best_res_dev['acc']*100,
                                                                                      best_res_dev['f1']*100,
                                                                                      best_res_dev['roc']*100,
                                                                                      best_res_dev['ap']*100))

                print('Best Test: Acc {0:.4}\tF1 {1:.4}\tAUC {2:.4}\tAP {3:.4}'.format(best_res_test['acc']*100,
                                                                                       best_res_test['f1']*100,
                                                                                       best_res_test['roc']*100,
                                                                                       best_res_test['ap']*100))

                print('Early stop at {0} epoch.'.format(epoch))
                break
