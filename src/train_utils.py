import os
import sys
import random
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import utils
import loader


def eval_link_context(dataset, model, criterion, args):
    with torch.no_grad():
        model.eval()

        test_logits = []
        test_labels = []
        num_batches = len(dataset) // args.batch_size
        np.random.shuffle(dataset)
        for i in range(num_batches):
            train_batch = dataset[i * args.batch_size: (i + 1) * args.batch_size]
            if i == num_batches - 1:
                train_batch = dataset[i * args.batch_size::]

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
            logits, _ = model(t1_batch, t2_batch, t1_ctx_batch, t2_ctx_batch, masks=masks)

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            test_logits.append(logits)
            test_labels.append(labels)

        test_golds = np.concatenate(test_labels)
        test_logits = np.concatenate(test_logits)
        test_probs = utils.sigmoid(test_logits)
        test_preds = np.where(utils.sigmoid(test_logits) >= 0.5, 1, 0)

        test_roc_auc = metrics.roc_auc_score(test_golds, test_probs)
        test_ap = metrics.average_precision_score(test_golds, test_probs)
        test_acc = metrics.accuracy_score(test_golds, test_preds)
        test_f1 = metrics.f1_score(test_golds, test_preds)
        # test_prec = metrics.precision_score(test_golds, test_preds)
        # test_reca = metrics.recall_score(test_golds, test_preds)

    return {'roc': test_roc_auc, 'ap': test_ap, 'acc': test_acc, 'f1': test_f1}


def logistis_regression(train, dev, test, embed_file, seed, link_feature='hadamard'):
    '''
    :param train:
    :param dev:
    :param test:
    :return:
    '''
    # construct X_train, y_train, X_test, y_test
    f = open(embed_file).readlines()
    print('Node Embeddings: ', f[0].strip())
    node_dict = {int(x.strip().split()[0]):
                     np.array(x.strip().split()[1::], dtype=np.float32) /
                     np.linalg.norm(np.array(x.strip().split()[1::], dtype=np.float32))
                 for x in f[1::]}

    if link_feature == 'concat':
        print('concat feature!')
        X_train = [np.append(node_dict[int(x[0])], node_dict[int(x[1])]) for x in train]
        X_dev = [np.append(node_dict[int(x[0])], node_dict[int(x[1])]) for x in dev]
        X_test = [np.append(node_dict[int(x[0])], node_dict[int(x[1])]) for x in test]
    elif link_feature == 'hadamard':
        print('hadamard feature!')
        X_train = [node_dict[int(x[0])] * node_dict[int(x[1])] for x in train]
        X_dev = [node_dict[int(x[0])] * node_dict[int(x[1])] for x in dev]
        X_test = [node_dict[int(x[0])] * node_dict[int(x[1])] for x in test]
    else:
        raise ValueError('Wrong link features!')

    y_train = [x[-1] for x in train]
    y_dev = [x[-1] for x in dev]
    y_test = [x[-1] for x in test]

    # shuffle for training and testing
    c = list(zip(X_train, y_train))
    np.random.shuffle(c)
    X_train, y_train = zip(*c)
    c = list(zip(X_dev, y_dev))
    # np.random.shuffle(c)
    X_dev, y_dev = zip(*c)
    c = list(zip(X_test, y_test))
    # random.shuffle(c)
    X_test, y_test = zip(*c)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_dev = np.array(X_dev)
    y_dev = np.array(y_dev)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf1 = LogisticRegression(random_state=seed, solver='lbfgs')
    clf1.fit(X_train, y_train)

    y_pred_proba = clf1.predict_proba(X_dev)[:, 1]
    # print(clf1.predict_proba(X_dev).shape)
    y_pred = clf1.predict(X_dev)
    auc_roc = metrics.roc_auc_score(y_dev, y_pred_proba)
    auc_pr = metrics.average_precision_score(y_dev, y_pred_proba)
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred)
    print('Dev: Acc {0:.4} | F1 {1:.4} | AUC {2:.4} | AP {3:.4}'.format(accuracy*100, f1*100, auc_roc*100, auc_pr*100))

    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = metrics.roc_auc_score(y_test, y_pred_proba)
    auc_pr = metrics.average_precision_score(y_test, y_pred_proba)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print('Test: Acc {0:.4} | F1 {1:.4} | AUC {2:.4} | AP {3:.4}'.format(accuracy*100, f1*100, auc_roc*100, auc_pr*100))
    return auc_roc, auc_pr, accuracy, f1


def eval_link_relation(dataloader, model, criterion, args):
    with torch.no_grad():
        model.eval()

        test_logits = []
        test_labels = []
        for heads, tails, labels in dataloader:

            heads, tails = heads.to(args.device), tails.to(args.device)
            labels = labels.type(torch.FloatTensor).to(args.device)
            # logits, _ = model(head_ctx, tail_ctx)
            logits = model(heads, tails)

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            test_logits.append(logits)
            test_labels.append(labels)

        test_golds = np.concatenate(test_labels)
        test_logits = np.concatenate(test_logits)
        test_probs = utils.sigmoid(test_logits)
        test_preds = np.where(utils.sigmoid(test_logits) >= 0.5, 1, 0)

        test_roc_auc = metrics.roc_auc_score(test_golds, test_probs)
        test_ap = metrics.average_precision_score(test_golds, test_probs)
        test_acc = metrics.accuracy_score(test_golds, test_preds)
        test_f1 = metrics.f1_score(test_golds, test_preds)
        # test_prec = metrics.precision_score(test_golds, test_preds)
        # test_reca = metrics.recall_score(test_golds, test_preds)

    return {'roc': test_roc_auc, 'ap': test_ap, 'acc': test_acc, 'f1': test_f1}
