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
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size*2)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([self.config.gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.config.hidden_size]),
            requires_grad=False
        )

    def init_weight(self):
        nn.init.uniform_(
            tensor=self.ent_embeddings,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, input, mode):
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

        if mode == 'head-batch':
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
        return score

    def loss(self, positive_score, negative_score, subsampling_weight=torch.tensor([1])):
        if self.config.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.config.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if self.config.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        if self.config.regularization != 0:
            # Use L3 regularization for ComplEx and DistMult
            normloss = self.config.regularization * (
                self.ent_embeddings.weight.data.norm(p=3)**3 +
                self.rel_embeddings.weight.data.norm(p=3).norm(p=3)**3
            )
            loss += normloss

        return loss, positive_sample_loss, negative_sample_loss

    def eval_model(self,input, mode):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        if mode=='head-batch':
            h = self.ent_embeddings.weight.data.repeat(len(h), 1, 1)
        else:
            t = self.ent_embeddings.weight.data.repeat(len(h), 1, 1)

        re_head, im_head = torch.chunk(h, 2, dim=2)
        re_tail, im_tail = torch.chunk(t, 2, dim=2)
        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = r / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
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
        if mode == 'head-batch':
            positive_arg = batch_h.squeeze()
        elif mode == 'tail-batch':
            positive_arg = batch_t.squeeze()
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

    def getWeights(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy()}