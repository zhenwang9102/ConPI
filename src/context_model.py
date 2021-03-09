import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextInteractionModel(nn.Module):
    def __init__(self, args):
        super(ContextInteractionModel, self).__init__()
        self.args = args
        self.embed_dim = args.node_embed_dim
        self.embed_size = args.node_vocab_size
        self.output_V = args.node_vocab_size
        self.dropout = nn.Dropout(args.dropout)

        self.context_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(args.pre_train_nodes),
                                                               freeze=args.freeze_node)

        if args.ctx_model == 'dynamic':
            self.linear_pred_3 = nn.Linear(self.embed_dim, 1)

        elif args.ctx_model == 'pairwise':
            self.pair_proj_layer = nn.Sequential(nn.Linear(3 * self.embed_dim, self.embed_dim),
                                                 nn.Tanh())
            self.out_layer = nn.Linear(self.embed_dim, 1)
        else:
            raise ValueError('Wrong model!')

        self.att_mat = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.att_mat)

    def _list_static_context(self, t1_contexts, t2_contexts):
        t1_context_vecs = []
        t2_context_vecs = []
        for i in range(len(t1_contexts)):
            t1_context = t1_contexts[i]
            if type(t1_context) != torch.Tensor:
                t1_context = torch.tensor(t1_context).to(self.args.device)
            t1_context = self.context_embeddings(t1_context)  # (W, dim for node embedding - D2)

            t2_context = t2_contexts[i]
            if type(t2_context) != torch.Tensor:
                t2_context = torch.tensor(t2_context).to(self.args.device)  # (context size for t2 - V, M2)
            t2_context = self.context_embeddings(t2_context)  # (V, D2)

            t1_context_vec = torch.mean(t1_context, dim=0, keepdim=True)  # (1, 128)
            t2_context_vec = torch.mean(t2_context, dim=0, keepdim=True)  # (1, 128)

            t1_context_vecs.append(t1_context_vec)
            t2_context_vecs.append(t2_context_vec)

        t1_context_vecs = torch.cat(t1_context_vecs, dim=0)
        t2_context_vecs = torch.cat(t2_context_vecs, dim=0)  # (N, word dim)
        return t1_context_vecs, t2_context_vecs

    def _list_dynamic_context(self, t1_contexts, t2_contexts):
        t1_context_vecs = []
        t2_context_vecs = []
        row_col_scores = []
        for i in range(len(t1_contexts)):
            t1_context = t1_contexts[i]
            if type(t1_context) != torch.Tensor:
                t1_context = torch.tensor(t1_context).to(self.args.device)
            t1_context = self.context_embeddings(t1_context)  # (W, dim for node embedding - D2)
            t1_context = t1_context / torch.norm(t1_context, p=2)

            t2_context = t2_contexts[i]
            if type(t2_context) != torch.Tensor:
                t2_context = torch.tensor(t2_context).to(self.args.device)  # (context size for t2 - V, M2)
            t2_context = self.context_embeddings(t2_context)  # (V, D2)
            t2_context = t2_context / torch.norm(t2_context, p=2)

            t1_context_vec, t2_context_vec, scores = self._dynamic_interaction(t1_context, t2_context)
            row_col_scores.append(scores)

            t1_context_vecs.append(t1_context_vec)
            t2_context_vecs.append(t2_context_vec)

        t1_context_vecs = torch.cat(t1_context_vecs, dim=0)
        t2_context_vecs = torch.cat(t2_context_vecs, dim=0)  # (N, word dim)
        return t1_context_vecs, t2_context_vecs, row_col_scores

    def _dynamic_interaction(self, mat_A, mat_B):
        # mat_A: (W, D), mat_B: (V, D)
        dim_A = mat_A.shape[0]
        dim_B = mat_B.shape[0]

        mat_sim = torch.matmul(torch.matmul(mat_A, self.att_mat), torch.t(mat_B))
        mat_sim = torch.tanh(mat_sim)

        rows = F.softmax(torch.mean(mat_sim, dim=1), dim=0).reshape(-1, 1)
        cols = F.softmax(torch.mean(mat_sim, dim=0), dim=0).reshape(-1, 1)

        new_A = torch.sum(mat_A * rows, dim=0, keepdim=True)
        new_B = torch.sum(mat_B * cols, dim=0, keepdim=True)
        return new_A, new_B, mat_sim

    def _list_pairwise_context(self, t1_contexts, t2_contexts):
        pair_vecs = []
        pair_scores = []
        for i in range(len(t1_contexts)):
            t1_context = t1_contexts[i]
            if type(t1_context) != torch.Tensor:
                t1_context = torch.tensor(t1_context).to(self.args.device)
            t1_context = self.context_embeddings(t1_context)  # (W, dim for node embedding - D2)
            t1_context = t1_context / torch.norm(t1_context, p=2)

            t2_context = t2_contexts[i]
            if type(t2_context) != torch.Tensor:
                t2_context = torch.tensor(t2_context).to(self.args.device)  # (context size for t2 - V, M2)
            t2_context = self.context_embeddings(t2_context)  # (V, D2)
            t2_context = t2_context / torch.norm(t2_context, p=2)

            # pairwise interaction
            pair_vec, pair_score = self._pairwise_interaction(t1_context, t2_context)
            pair_vecs.append(pair_vec)
            pair_scores.append(pair_score)

        pair_vecs = torch.cat(pair_vecs, dim=0)
        return pair_vecs, pair_scores

    def _pairwise_interaction(self, mat_A, mat_B):
        # mat_A: (W, D), mat_B: (V, D)
        dim_A = mat_A.shape[0]
        dim_B = mat_B.shape[0]

        pair_scores = torch.matmul(torch.matmul(mat_A, self.att_mat), torch.t(mat_B))  # [W, V]
        pair_scores = pair_scores.reshape(-1)
        norm_scores = F.softmax(pair_scores, dim=0)  # [W*V]

        repeat_A = mat_A.repeat(1, dim_B).reshape(dim_A * dim_B, -1)  # [W*V, D]
        repeat_B = mat_B.repeat(dim_A, 1)  # [W*V, D]

        pair_vec = torch.cat((repeat_A, repeat_B, repeat_A * repeat_B), dim=-1)  # [W*V, 2*D]
        pair_fuse = torch.matmul(norm_scores.unsqueeze(0), pair_vec) # [2*D]

        return pair_fuse, norm_scores

    def forward(self, t1s, t2s, t1_contexts, t2_contexts, masks=None):
        # list of pairs: (head_ctx, tail_ctx)
        t1_embed = self.dropout(self.context_embeddings(t1s))
        t2_embed = self.dropout(self.context_embeddings(t2s))

        # context interaction
        if self.args.ctx_model == 'static':
            ctx_t1s, ctx_t2s = self._list_static_context(t1_contexts, t2_contexts)
            ctx_score = self.linear_pred(torch.cat((ctx_t1s, ctx_t2s), dim=1)).reshape(-1)
            logits = ctx_score
            pair_scores = None

        elif self.args.ctx_model == 'dynamic':
            ctx_t1s, ctx_t2s, pair_scores = self._list_dynamic_context(t1_contexts, t2_contexts)
            ctx_score = self.linear_pred_3(ctx_t1s * ctx_t2s).reshape(-1)
            logits = ctx_score

        else:
            pair_vecs, pair_scores = self._list_pairwise_context(t1_contexts, t2_contexts)
            pair_vecs = self.pair_proj_layer(pair_vecs)
            logits = self.out_layer(pair_vecs).reshape(-1)

        return logits, pair_scores
