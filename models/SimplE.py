import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimplE(nn.Module):
    def __init__(self, config):
        super(SimplE, self).__init__()
        self.config = config
        self.ent_h_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_t_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_inv_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()

    def init_weight(self):
        nn.init.xavier_uniform_(self.ent_h_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_t_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight.data)

    def norm_loss(self):
        return ((torch.norm(self.ent_h_embeddings.weight, p=self.config.L) ** 2) / self.config.entTotal + (
                torch.norm(self.ent_t_embeddings.weight, p=self.config.L) ** 2) / self.config.entTotal + (
                        torch.norm(self.rel_embeddings.weight, p=self.config.L) ** 2) / self.config.relTotal + (
                        torch.norm(self.rel_inv_embeddings.weight, p=self.config.L) ** 2) / self.config.relTotal) / 2

    def forward(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        hh_embs = self.ent_h_embeddings(batch_h).squeeze(1)
        ht_embs = self.ent_h_embeddings(batch_t).squeeze(1)
        th_embs = self.ent_t_embeddings(batch_h).squeeze(1)
        tt_embs = self.ent_t_embeddings(batch_t).squeeze(1)
        r_embs = self.rel_embeddings(batch_r).squeeze(1)
        r_inv_embs = self.rel_inv_embeddings(batch_r).squeeze(1)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        score = torch.clamp((scores1 + scores2) / 2, -20, 20)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        score_loss = torch.sum(self.criterion(n_score - p_score))
        norm_loss = self.norm_loss() * self.config.regularization
        return score_loss + norm_loss

    def get_positive_score(self, score):
        return score[0:self.config.batch_size]

    def get_negative_score(self, score):
        return score[self.config.batch_size:self.config.batch_seq_size]

    def getWeights(self):
        return {"ent_h_embeddings": self.ent_h_embeddings.weight.detach().cpu().numpy(),
                "ent_t_embeddings": self.ent_t_embeddings.weight.detach().cpu().numpy(),
                "rel_embeddings": self.rel_embeddings.weight.detach().cpu().numpy(),
                "rel_inv_embeddings": self.rel_inv_embeddings.weight.detach().cpu().numpy()}

    def eval_model(self, input):
        batch_h, batch_r, batch_t = torch.chunk(input=input, chunks=3, dim=1)
        hh_embs = self.ent_h_embeddings(batch_h).squeeze(1)
        ht_embs = self.ent_h_embeddings(batch_t).squeeze(1)
        th_embs = self.ent_t_embeddings(batch_h).squeeze(1)
        tt_embs = self.ent_t_embeddings(batch_t).squeeze(1)
        r_embs = self.rel_embeddings(batch_r).squeeze(1)
        r_inv_embs = self.rel_inv_embeddings(batch_r).squeeze(1)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        target_score = torch.clamp((scores1 + scores2) / 2, -20, 20)

        #predict head
        scores1 = torch.sum(self.ent_h_embeddings.weight.data*r_embs*tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * self.ent_t_embeddings.weight.data, dim=1)
        head_score = torch.clamp((scores1 + scores2) / 2, -20, 20)

        #predict tail
        scores1 = torch.sum(hh_embs * r_embs * self.ent_t_embeddings.weight.data, dim=1)
        scores2 = torch.sum(self.ent_h_embeddings.weight.data * r_inv_embs * th_embs, dim=1)
        tail_score = torch.clamp((scores1 + scores2) / 2, -20, 20)

        rankH = torch.nonzero(nn.functional.relu(head_score - target_score)).size()[0]
        rankT = torch.nonzero(nn.functional.relu(tail_score - target_score)).size()[0]

        return rankH + 1, rankT + 1


