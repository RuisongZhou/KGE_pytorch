import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import codecs
from utils.loss import normLoss


class TransD(nn.Module):
    def __init__(self, config, L=1):
        super(TransD, self).__init__()
        self.config = config
        self.L = L
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.distfn = nn.PairwiseDistance(L)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        h = torch.squeeze(h, dim=1)
        r = torch.squeeze(r, dim=1)
        t = torch.squeeze(t, dim=1)
        return self.distfn(h + r, t)

    def _transfer(self, e, e_transfer, r_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * r_transfer
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

    def loss(self, p_score, n_score):
        if self.config.usegpu:
            y = Variable(torch.Tensor([self.config.margin]).cuda())
        else:
            y = Variable(torch.Tensor([self.config.margin]))
        return torch.sum(F.relu(input=p_score - n_score + y)) / self.config.batch_size

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        normlosses = self.norm_loss() * self.config.regularization
        return self.loss(p_score, n_score) + normlosses

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
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)

        '''
        ent_embeddings.shape = (ent, hidden_size)
        mutmat.shape = (ent,1)
        emb_ent.shape = (ent, hidden_size)
        '''
        emb_ent = self._transfer(self.ent_embeddings.weight.data, self.ent_transfer.weight.data,
                                 r_transfer.squeeze(dim=1))

        targetLoss = torch.norm(h + r - t, self.L)

        tmpHeadEmbedding = h.squeeze(dim=1)
        tmpRelationEmbedding = r.squeeze(dim=1)
        tmpTailEmbedding = t.squeeze(dim=1)

        tmpHeadLoss = torch.norm(emb_ent + tmpRelationEmbedding - tmpTailEmbedding,
                                 self.L, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - emb_ent,
                                 self.L, 1).view(-1, 1)

        rankH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss)).size()[0]
        rankT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss)).size()[0]

        return rankH + 1, rankT + 1

    def getWeights(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy(),
                "entityTransfer": self.ent_transfer.weight.detach().cpu().numpy(),
                "relationTransfer": self.rel_transfer.weight.detach().cpu().numpy()}

    def norm_loss(self):
        loss_emb = torch.norm(self.ent_embeddings.weight.data, p=self.config.L) / self.config.entTotal + torch.norm(
            self.rel_embeddings.weight.data, p=self.config.L) / self.config.relTotal
        loss_transfer = torch.norm(self.ent_transfer.weight.data, p=self.config.L) / self.config.entTotal + torch.norm(
            self.rel_transfer.weight.data, p=self.config.L) / self.config.relTotal

        return loss_emb + loss_transfer
