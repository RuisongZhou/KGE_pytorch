import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import codecs
from utils.loss import normLoss


class TransH(nn.Module):
    def __init__(self, config):
        super(TransH, self).__init__()
        self.config = config

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.norm_vector = nn.Embedding(self.config.relTotal, self.config.hidden_size)  # Hyperplane Matrix
        self.distfn = nn.PairwiseDistance(self.config.L)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _calc(self, h, t, r):
        h = torch.squeeze(h, dim=1)
        r = torch.squeeze(r, dim=1)
        t = torch.squeeze(t, dim=1)
        return self.distfn(h + r, t)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def loss(self, p_score, n_score):
        if self.config.usegpu:
            y = Variable(torch.Tensor([self.config.margin]).cuda())
        else:
            y = Variable(torch.Tensor([self.config.margin]))

        marginloss = torch.sum(F.relu(input=p_score - n_score + y)) / self.config.batch_size
        entityloss = torch.sum(F.relu(torch.norm(self.ent_embeddings.weight, p=2, dim=1, keepdim=False) - 1))
        orthLoss = torch.sum(
            F.relu(torch.sum(self.norm_vector.weight * self.rel_embeddings.weight, dim=1, keepdim=False) / \
                   torch.norm(self.rel_embeddings.weight, p=2, dim=1, keepdim=False) - self.config.eps ** 2))

        return marginloss + self.config.C * (entityloss / self.config.entTotal + orthLoss / self.config.relTotal)

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_norm = self.norm_vector(batch_r)
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)

        return self.loss(p_score, n_score)

    def get_positive_score(self, score):
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        negative_score = score[self.config.batch_size:self.config.batch_seq_size]
        return negative_score

    def eval_model(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_norm = self.norm_vector(batch_r)
        h = self._transfer(h, r_norm)
        t = self._transfer(t, r_norm)

        ent_emb = self._transfer(self.ent_embeddings.weight.data, r_norm.squeeze(dim=1))

        targetLoss = torch.norm(h + r - t, self.config.L)

        tmpHeadEmbedding = h.squeeze(dim=1)
        tmpRelationEmbedding = r.squeeze(dim=1)
        tmpTailEmbedding = t.squeeze(dim=1)

        tmpHeadLoss = torch.norm(ent_emb + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.config.L, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - ent_emb,
                                 self.config.L, 1).view(-1, 1)

        rankH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss)).size()[0]
        rankT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss)).size()[0]

        return rankH + 1, rankT + 1

    def getWeights(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy(),
                "norm_vector": self.norm_vector.weight.detach().cpu().numpy()}
