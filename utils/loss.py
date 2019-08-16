import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


def normLoss(embeddings, dim=1):
    if torch.cuda.is_available():
        floatTensor = torch.cuda.FloatTensor
    else:
        floatTensor = torch.FloatTensor

    norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))
