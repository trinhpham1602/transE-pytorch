import numpy as np
import torch
import torch.nn as nn
import os
import GANs


class NoiAwareKGE(nn.Module):

    def __init__(self, pretrain_entities_emb, pretrain_relations_emb, emb_dim, device, norm=1, margin=1.0):
        super(NoiAwareKGE, self).__init__()
        self.device = device
        self.norm = norm
        self.emb_dim = emb_dim
        self.margin = margin
        self.entities_emb = pretrain_entities_emb  # type nn.Embedding
        self.relations_emb = pretrain_relations_emb  # type nn.Embedding

    def _distance(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
                                                                                                          dim=1)

    def _get_emb(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return torch.cat((self.entities_emb(heads), self.relations_emb(relations), self.entities_emb(tails)), dim=1)

    def predict(self, triplets: torch.LongTensor):
        return self._distance(triplets)

    def forward(self, positive_triples, negative_triples, D: GANs.Discriminator, G: GANs.Generator):
        # G take hrt concat
        distance_pos_triples = self._distance(positive_triples)
        positive_triples = self._get_emb(positive_triples)
        positive_triples = torch.reshape(
            positive_triples, (len(positive_triples), 3, self.emb_dim))
        negative_triples = self._get_emb(negative_triples)

        true_neg_triples = negative_triples[G.forward(negative_triples)[
            :, 0] > 0.5]
        true_neg_triples = torch.reshape(
            true_neg_triples, (len(true_neg_triples), 3, self.emb_dim))

        distance_neg_triples = (
            true_neg_triples[:, 0] + true_neg_triples[:, 1] - true_neg_triples[:, 2]).norm(p=self.norm, dim=1)

        pos_scores = - \
            torch.log(torch.sigmoid(self.margin -
                                    distance_pos_triples))
        neg_scores = 1/len(true_neg_triples)*torch.sum(distance_neg_triples)
        sum_scores = torch.sum(D.forward(positive_triples)
                               * (pos_scores + neg_scores))
        return sum_scores
