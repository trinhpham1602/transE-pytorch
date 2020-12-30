import numpy as np
import torch
import torch.nn as nn
import os


class NoiAwareKGE(nn.Module):

    def __init__(self, n_samples, device, emb_dim, norm=1, margin=1.0):
        super().__init__()
        self.n_samples = n_samples
        self.norm = norm
        self.device = device
        self.emb_dim = emb_dim
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.W = self._init_W()

    def _init_W(self):
        W = nn.Embedding(num_embeddings=self.n_samples,
                         embedding_dim=self.emb_dim * 3,  # concat(h,r,t)
                         )
        uniform_range = 6 / np.sqrt(self.emb_dim*3)
        W.weight.data.uniform_(-uniform_range, uniform_range)
        return W

    def forward(self, pos_triples, neg_triples, order_hrt):

        pos_triples_concat = torch.reshape(
            pos_triples, (len(pos_triples), self.emb_dim*3))
        new_pos_concat = self.W(
            order_hrt)*pos_triples_concat
        new_pos_triples = torch.reshape(
            new_pos_concat, (len(pos_triples), 3, self.emb_dim))
        return self.loss(new_pos_triples, neg_triples)

    def loss(self, pos_triples, neg_triples):
        distance_pos = torch.sum(pos_triples, dim=1).norm(p=self.norm, dim=1)
        print(distance_pos.size())
        distance_neg = torch.sum(neg_triples, dim=1).norm(p=self.norm, dim=1)
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(distance_pos, distance_neg, target)
