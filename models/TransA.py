import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TransA(nn.Module):
    def __init__(self, config):
        super(TransA, self).__init__()
        self.config = config

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.init_weight()
        self.init_Wr()

    def init_weight(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def init_Wr(self):
        if self.config.usegpu:
            self.Wr = torch.zeros((self.config.relTotal, self.config.hidden_size, self.config.hidden_size),requires_grad=False).cuda()
        else:
            self.Wr = torch.zeros((self.config.relTotal, self.config.hidden_size, self.config.hidden_size),requires_grad=False)

    def _calc_Wr(self, h, r, t):
        '''
        Calculate the Mahalanobis distance weights
        Wr.shape = (batch_size, hidden_size, hidden_size)
        '''
        error = torch.abs(h + r - t)
        pos_error = error[0:self.config.batch_size]
        neg_error = error[self.config.batch_size:self.config.batch_seq_size]
        return torch.matmul(neg_error.permute((0, 2, 1)), neg_error) - torch.matmul(pos_error.permute((0, 2, 1)),
                                                                                    pos_error)

    def _calc_(self, h, r, t, Wr):
        '''
        calculate the score loss
        '''
        error = torch.abs(h + r - t)

        p_error = torch.matmul(
            torch.matmul(error[0:self.config.batch_size], torch.unsqueeze(Wr, dim=0)),
            error[0:self.config.batch_size].transpose(1, 2))
        n_error = torch.matmul(
            torch.matmul(error[self.config.batch_size:self.config.batch_seq_size], torch.unsqueeze(Wr, dim=0)),
            error[self.config.batch_size:self.config.batch_seq_size].transpose(1, 2))
        return torch.squeeze(p_error), torch.squeeze(n_error)

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        pos_batch_r = batch_r[0:self.config.batch_size].unsqueeze(1).repeat(1, self.config.hidden_size,
                                                                            self.config.hidden_size)
        relWr = self._calc_Wr(h, r, t)

        self.Wr.scatter_(0, pos_batch_r, relWr)
        p_score, n_score = self._calc_(h, r, t, relWr)

        marginLoss = torch.sum(F.relu(input=p_score - n_score + self.config.margin)) / self.config.batch_size
        Wrloss = torch.norm(relWr, p=self.config.L) * self.config.lamb
        normloss = self.norm_loss(h, r, t) * self.config.regularization
        return marginLoss + Wrloss + normloss

    def norm_loss(self, h, r, t):
        loss = h.norm(self.config.L) + r.norm(self.config.L) + t.norm(self.config.L)
        return loss

    def eval_model(self, input):
        '''
        batch_size should be 1
        '''
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        relWr = self.Wr[batch_r]

        # predict tail
        error = torch.abs(h + r - self.ent_embeddings.weight.data)
        score = torch.matmul(torch.matmul(error, relWr), error.transpose(1, 2)).squeeze(1)
        right = torch.abs(h+r-t)
        targetloss = torch.matmul(torch.matmul(right, relWr), error.transpose(1, 2)).squeeze(1)
        rankT = torch.nonzero(nn.functional.relu(score - targetloss)).size()[0]

        #predict head
        error = torch.abs(self.ent_embeddings.weight.data + r -t )
        score = torch.matmul(torch.matmul(error, relWr), error.transpose(1, 2)).squeeze(1)
        rankH = torch.nonzero(nn.functional.relu(score - targetloss)).size()[0]

        return rankH + 1, rankT + 1

    def getWeights(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy(),
                "Wr": self.Wr.cpu().numpy()}
