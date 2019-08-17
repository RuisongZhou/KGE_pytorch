import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import numpy as np


class ConvE(nn.Module):
    def __init__(self, config):
        super(ConvE, self).__init__()
        self.config = config

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)  # must be 200
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.inp_drop = torch.nn.Dropout(config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(config.hidden_size)
        self.register_parameter('b', Parameter(torch.zeros(self.config.entTotal)))
        self.fc = torch.nn.Linear(10368, config.hidden_size)

        self.init_weights()

    def init_weights(self):
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h).view(-1, 1, 10, 20)
        r = self.rel_embeddings(batch_r).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([h, r], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(self.config.batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        if self.config.usegpu:
            one_hot = torch.zeros(self.config.batch_size, self.config.entTotal).scatter_(1, batch_t.cpu(), 1).cuda()
        else:
            one_hot = torch.zeros(self.config.batch_size, self.config.entTotal).scatter_(1, batch_t.cpu(), 1)

        one_hot = ((1.0 - self.config.label_smoothing_epsilon) * one_hot) + (1.0 / one_hot.size(1))
        loss = self.loss(pred, one_hot)

        return loss

    def eval_model(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        h = self.ent_embeddings(batch_h).view(-1, 1, 10, 20)
        r = self.rel_embeddings(batch_r).view(-1, 1, 10, 20)
        size = len(batch_t)
        stacked_inputs = torch.cat([h, r], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(size, -1)
        # print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        maxvalues, argsort = torch.sort(pred, dim=1, descending=True)

        batch_t = batch_t.cpu().numpy()
        argsort = argsort.cpu().numpy()

        ranks = []
        hit10 = 0
        hit1 = 0
        for i in range(len(batch_t)):
            rank = np.where(argsort[i] == batch_t[i, 0].item())[0][0]
            ranks.append(rank)
            if rank < 10:
                hit10 += 1
            if rank < 1:
                hit1 += 1

        rank = sum(ranks) / len(batch_t)
        hit10 /= len(batch_t)
        hit1 /= len(batch_t)

        return rank, hit10, hit1

    def getWeight(self):
        return {"entityEmbed": self.ent_embeddings.weight.detach().cpu().numpy(),
                "relationEmbed": self.rel_embeddings.weight.detach().cpu().numpy()}

