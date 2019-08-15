import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import codecs

pi = 3.14159265358979323846

class RotatE(nn.Module):
    def __init__(self, config):
        super(RotatE, self).__init__()
        self.config = config
        self.mode = config.mode
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size*2)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size*2)

        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([self.config.gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.config.hidden_size]),
            requires_grad=False
        )


    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        re_head, im_head = torch.chunk(h, 2, dim=2)
        re_tail, im_tail = torch.chunk(t, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = r / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if self.mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        pos_re = self.get_positive_score(re_score)
        neg_re = self.get_negative_score(re_score)
        pos_im = self.get_positive_score(im_score)
        neg_im = self.get_negative_score(im_score)


        p_score = torch.stack([pos_re, pos_im], dim=0)
        p_score = p_score.norm(dim=0)
        p_score = self.gamma.item() - p_score.sum(dim=2)

        n_score = torch.stack([neg_re,neg_im], dim=0)
        n_score = n_score.norm(dim=0)
        n_score = self.gamma.item() - n_score.sum(dim=2)
        return p_score, n_score


    def get_positive_score(self, score):
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        negative_score = score[self.config.batch_size:self.config.batch_seq_size]
        return negative_score


    def eval_model(self,input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        re_head, im_head = torch.chunk(h, 2, dim=2)
        re_tail, im_tail = torch.chunk(t, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = r / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if self.mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)

        argsort = torch.argsort(score, dim=1, descending=True)
        if self.mode == 'head-batch':
            positive_arg = batch_h
        elif self.mode == 'tail-batch':
            positive_arg = batch_t
        else:
            raise ValueError('mode %s not supported' % self.mode)

        ranks = 0
        hit1 = 0
        hit10 = 0
        for i in range(len(h)):
            # Notice that argsort is not ranking
            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
            assert ranking.size(0) == 1
            # ranking + 1 is the true ranking used in evaluation metrics
            ranking = 1 + ranking.item()
            ranks +=ranking
            hit1 += 1 if ranking <=1 else 0
            hit10 += 1 if ranking <=10 else 0

        ranks /= len(h)
        hit1 /= len(h)
        hit10 /= len(h)

        return ranks, hit1, hit10