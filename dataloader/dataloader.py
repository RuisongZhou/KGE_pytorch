import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import Config
import math


class UniformDataSet(Dataset):
    '''
    the negative samples are subject to uniform distribution
    '''

    def __init__(self, datapath, neg_sample_rate=1, rephead=0.5, reptail=0.5):
        super(Dataset, self).__init__()
        self.dataPath = datapath
        print("[INFO] Load knowledge graph embedding data")
        self.data = pd.read_csv(datapath,
                                sep='\t',
                                names=["head", "tail", "relation"],
                                header=None,
                                )
        order = ['head', 'relation', 'tail']
        self.data = self.data[order]
        self.generate_neg_samples(rephead=rephead, reptail=reptail, neg_sample_rate=neg_sample_rate)
        print("[INFO] Finish load knowledge graph embedding data")

    def __len__(self):
        return len(self.posData)

    def __getitem__(self, item):
        return self.posData[item], self.negData[item]

    # [TODO] Use KBGAN
    def generate_neg_samples(self, rephead, reptail, neg_sample_rate=1):
        '''
        Use to generate negative sample for training
        @param
        rephead: (float)Probability of replacing head
        reptail: (fload)Probability of replacing head with tail entities or replacing tail with head entities.
        '''
        assert rephead >= 0 and rephead < 1 and reptail >= 0 and reptail <= 1
        print("[INFO] Generate negtive samples from positive samples.")
        self.negData = self.data.copy()
        np.random.seed(0)
        repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negData),))
        exProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negData),))
        shuffleHead = self.negData["head"].sample(frac=1.0, random_state=0)
        shuffleTail = self.negData["tail"].sample(frac=1.0, random_state=0)

        # Replacing head or tail
        def replaceHead(relHead, shuffHead, shuffTail, repP, exP):
            if repP >= rephead:
                # Not replacing head.self.negD
                return relHead
            else:
                if exP > reptail:
                    # Replacing head with shuffle head.
                    return shuffHead
                else:
                    # Replacing head with shuffle tail.
                    return shuffTail

        def replaceTail(relTail, shuffHead, shuffTail, repP, exP):
            if repP < rephead:
                # Not replacing tail.
                return relTail
            else:
                if exP > relTail:
                    # Replacing tail with shuffle tail.
                    return shuffTail
                else:
                    #  Replacing head with shuffle head.
                    return shuffHead

        self.negData["head"] = list(
            map(replaceHead, self.negData["head"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))
        self.negData["tail"] = list(
            map(replaceTail, self.negData["tail"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))
        self.negData = np.array(self.negData)
        self.negData = self.negData[:,np.newaxis,:]
        self.posData = self.data.copy()
        self.posData = np.array(self.posData)
        self.posData = self.posData[:,np.newaxis,:]

        if neg_sample_rate > 1:
            self.pos_new_data = self.posData.copy()
            for i in range(neg_sample_rate-1):
                np.random.seed(i+1)
                self.exData = self.data.copy()
                repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.exData),))
                exProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.exData),))
                shuffleHead = self.exData["head"].sample(frac=1.0, random_state=0)
                shuffleTail = self.exData["tail"].sample(frac=1.0, random_state=0)
                self.exData["head"] = list(
                    map(replaceHead, self.exData["head"], shuffleHead, shuffleTail, repProbaDistribution,
                        exProbaDistribution))
                self.exData["tail"] = list(
                    map(replaceTail, self.exData["tail"], shuffleHead, shuffleTail, repProbaDistribution,
                        exProbaDistribution))

                self.exData = np.array(self.exData)
                self.exData = self.exData[:,np.newaxis,:]
                self.negData = np.concatenate((self.negData, self.exData),axis=1)
                self.posData = np.concatenate((self.posData, self.pos_new_data), axis=1)
                del self.exData

            del self.pos_new_data
        del self.data

class AdversarialDataset(Dataset):
    '''
    This negative sample generate method is come from RotatE.
    They sample negative triples from the following distribution:
    p\left(h_{j}^{\prime}, r, t_{j}^{\prime} |\left\{\left(h_{i}, r_{i}, t_{i}\right)\right\}\right)=\frac{\exp \alpha f_{r}\left(\mathbf{h}_{j}^{\prime},
     \mathbf{t}_{j}^{\prime}\right)}{\sum_{i} \exp \alpha f_{r}\left(\mathbf{h}_{i}^{\prime}, \mathbf{t}_{i}^{\prime}\right)}
    '''

    def __init__(self, datapath, ent_num, mode='tail-batch'):
        super(Dataset, self).__init__()
        self.dataPath = datapath
        self.ent_num = ent_num
        self.mode = mode
        print("[INFO] Load knowledge graph embedding data")
        self.data = self.get_data()
        self.count = self.count_frequency(self.data)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.data)
        print("[INFO] Finish load knowledge graph embedding data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos_sample = self.data[idx]
        head, relation, tail = pos_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        neg_sample = self.get_neg_sample(pos_sample)
        neg_head = neg_sample[0]
        neg_tail = neg_sample[1]
        if self.mode == 'tail_batch':
            neg_sample = (head, relation, neg_tail)
        else:
            neg_sample = (neg_head, relation, tail)
        pos_sample = torch.LongTensor(pos_sample)
        neg_sample = torch.LongTensor(neg_sample)
        return pos_sample, neg_sample, subsampling_weight, self.mode

    def get_neg_sample(self, sample):
        head, relation, tail = sample
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < 2:
            negative_sample = np.random.randint(self.ent_num, size=2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:2]
        negative_sample = torch.from_numpy(negative_sample)
        return negative_sample

    def get_data(self):
        tirples = []
        with open(self.dataPath) as f:
            for line in f:
                h, t, r = line.strip().split('\t')
                tirples.append([h, r, t])
            tirples = [tuple(map(int, i)) for i in tirples]
        return tirples

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, datapath, mode='tail-batch'):
        super(TestDataset, self).__init__()
        self.dataPath = datapath
        self.mode = mode
        print("[INFO] Load knowledge graph embedding data")
        self.data = self.get_data()
        print("[INFO] Finish load knowledge graph embedding data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos_sample = self.data[idx]
        return pos_sample

    def get_data(self):
        triples = pd.read_csv(self.dataPath,
                              sep='\t',
                              names=["head", "tail", "relation"],
                              header=None,
                              )
        order = ['head', 'relation', 'tail']
        triples = triples[order]
        return np.array(triples)


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        self.length = len(dataloader_head) * 2

    def __next__(self):
        self.step += 1
        if self.step >= self.length:
            raise StopIteration
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


if __name__ == '__main__':
    path = './data/FB15k/train2id.txt'
    # dataset = EmDataSet(path)
    # dataloader = DataLoader(dataset,
    #                         batch_size=32,
    #                         shuffle=True,
    #                         num_workers=8,
    #                         drop_last=False)
    # print("Finish make dataloader")
    # for each in enumerate(dataloader):
    #     print(each)
    #     break
    config = Config.Config()
    config.init()
    dataset = AdversarialDataset(path, config.modelparam.entTotal)
    dataloader = DataLoader(dataset, batch_size=10, num_workers=8)
    print("Finish make dataloader")
    # print(len(dataloader))
    # for each in dataloader:
    #     print(each[0])
    #     print(each[1])
    #     print(each[2])
    #     print(each[3])
    #     break

    dataloader2 = DataLoader(AdversarialDataset(path, config.modelparam.entTotal, mode='head-batch'),
                             batch_size=10, num_workers=8)

    train_iterator = BidirectionalOneShotIterator(dataloader, dataloader2)
    print(next(train_iterator))
