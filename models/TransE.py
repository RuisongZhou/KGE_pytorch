import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import codecs


class TransE(nn.Module):
    def __init__(self, config, ):
        super(TransE, self).__init__()
        self.config = config
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.distfn = nn.PairwiseDistance(self.config.L)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        h = torch.squeeze(h, dim=1)
        r = torch.squeeze(r, dim=1)
        t = torch.squeeze(t, dim=1)
        return self.distfn(h + r, t)

    def score_loss(self, p_score, n_score):
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
        score = self._calc(h, t, r)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        normloss = self.norm_loss() * self.config.regularization
        return self.score_loss(p_score, n_score) + normloss

    def predict(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()

    def get_positive_score(self, score):
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        negative_score = score[self.config.batch_size:self.config.batch_seq_size]
        return negative_score

    def load_emb_weight(self, entityEmbedFile, relationEmbedFile):
        print("[INFO] Loading entity pre-training embedding.")
        with codecs.open(entityEmbedFile, "r", encoding="utf-8") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.entityEmbedding.weight.size()[-1]
            for line in fp:
                ent, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                self.ent_embeddings.weight.data[ent].copy_(torch.from_numpy(embed))
        print("[INFO] Loading relation pre-training embedding.")
        with codecs.open(relationEmbedFile, "r", encoding="utf-8") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.relationEmbedding.weight.size()[-1]
            for line in fp:
                rel, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                self.rel_embeddings.weight.data[rel].copy_(torch.from_numpy(embed))

    def getWeights(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy()}

    def eval_model(self, input):
        norm = 2
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        # targetLoss = torch.norm(h + r - t, norm).repeat(self.config.entTotal, 1)
        # tmpHeadEmbedding = h.squeeze().repeat(self.config.entTotal, 1)
        # tmpRelationEmbedding = r.squeeze().repeat(self.config.entTotal, 1)
        # tmpTailEmbedding = t.squeeze().repeat(self.config.entTotal, 1)
        targetLoss = torch.norm(h + r - t, norm)
        tmpHeadEmbedding = h.squeeze(dim=1)
        tmpRelationEmbedding = r.squeeze(dim=1)
        tmpTailEmbedding = t.squeeze(dim=1)

        tmpHeadLoss = torch.norm(self.ent_embeddings.weight.data + tmpRelationEmbedding - tmpTailEmbedding,
                                 norm, 1).view(-1, 1)
        tmpTailLoss = torch.norm(tmpHeadEmbedding + tmpRelationEmbedding - self.ent_embeddings.weight.data,
                                 norm, 1).view(-1, 1)

        rankH = torch.nonzero(nn.functional.relu(targetLoss - tmpHeadLoss)).size()[0]
        rankT = torch.nonzero(nn.functional.relu(targetLoss - tmpTailLoss)).size()[0]

        return rankH + 1, rankT + 1

    def norm_loss(self):
        loss = torch.norm(self.ent_embeddings.weight.data, p=self.config.L) / self.config.entTotal + torch.norm(
            self.rel_embeddings.weight.data, p=self.config.L) / self.config.relTotal
        return loss
