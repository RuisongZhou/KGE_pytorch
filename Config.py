# coding:utf-8
import os
import torch
from torch.autograd import Variable
import os
import pandas as pd
import multiprocessing
from utils.util import CheckPath


class Config():
    def __init__(self):

        # set path and savetype
        self.dataset = 'FB15k'
        self.rootpath = './data/' + self.dataset
        self.trainpath = self.rootpath + "/train2id.txt"
        self.testpath = self.rootpath + '/test2id.txt'
        self.validpath = self.rootpath + "/valid2id.txt"
        self.entitypath = self.rootpath + '/entity2id.txt'
        self.relationpath = self.rootpath + '/relation2id.txt'

        # Other arguments
        self.embedpath = "./source/embed/"
        self.logpath = "./source/log/"
        self.modelpath = "./source/model/"
        self.summarydir = "./source/summary/"

        # DataLoader arguments
        self.batch_size = 1024
        self.eval_batch_size = 1
        self.numworkers = 6
        self.evalnumberworkers = multiprocessing.cpu_count()
        self.neg_sample_rate = 1

        # Model arguments
        self.model = 'TransD'
        self.usegpu = torch.cuda.is_available()
        self.epochs = 500
        self.evalepoch = 5
        self.learningrate = 0.01
        self.lrdecay = 0.95
        self.lrdecayepoch = 5
        self.shuffle = True
        self.optimizer = 'Adam'
        self.weight_decay = 0
        self.drop_last = True

        # load pre-trained model/emb
        self.init_checkpoint = None
        self.entityfile = "./source/embed/entityEmbedding.txt"
        self.relationfile = "./source/embed/relationEmbedding.txt"
        self.loadembed = False

        # Check Path
        self.check()

    def init(self):

        if self.model == 'TransE':
            self.modelparam = TransE_config()
        elif self.model == 'TransD':
            self.modelparam = TransD_config()
        elif self.model == 'TransH':
            self.modelparam = TransH_config()
        elif self.model == 'TransA':
            self.modelparam = TransA_config()
        elif self.model == 'ConvE':
            self.modelparam = ConvE_config()
        elif self.model == 'RotatE':
            self.modelparam = RotatE_config()
        elif self.model == 'SimplE':
            self.modelparam = SimplE_config()
        else:
            print("error model name %s" %self.model)
            exit(1)

        self.modelparam.entTotal = len(pd.read_csv(self.entitypath, sep='\t', header=None))
        self.modelparam.relTotal = len(pd.read_csv(self.relationpath, sep='\t', header=None))
        self.modelparam.batch_size = self.batch_size * self.neg_sample_rate
        self.modelparam.batch_seq_size = int(self.batch_size * 2)
        self.modelparam.usegpu = self.usegpu

        # save path
        self.savepath = self.modelpath + self.model
        CheckPath(self.savepath, raise_error=False)
    def check(self):
        # Check files
        CheckPath(self.trainpath)
        CheckPath(self.testpath)
        CheckPath(self.validpath)

        # Check dirs
        CheckPath(self.modelpath, raise_error=False)
        CheckPath(self.summarydir, raise_error=False)
        CheckPath(self.logpath, raise_error=False)
        CheckPath(self.embedpath, raise_error=False)



    def set_model_arg(self, lr=0.01, lrdecay=0.96, lrdecayepoch=5, epoch=5, hidden_size=100, margin=1, shuffle=True,
                      optimizer='Adam'):
        self.learningrate = lr
        self.lrdecay = lrdecay
        self.lrdecayepoch = lrdecayepoch
        self.epochs = epoch
        self.hidden_size = hidden_size
        self.margin = margin
        self.shuffle = shuffle
        self.optimizer = optimizer

    def set_setpath(self, trainpath=None, testpath=None, validpath=None, entitypath=None, \
                    relationpath=None, summarydir=None, modelpath=None):
        if trainpath:
            self.trainpath = trainpath
        if testpath:
            self.testpath = testpath
        if validpath:
            self.validpath = validpath
        if entitypath:
            self.entitypath = entitypath
            self.init()
        if relationpath:
            self.relationpath = relationpath
            self.init()
        if summarydir:
            self.summarydir = summarydir
        if modelpath:
            self.modelpath = modelpath
        self.check()

    def set_dataloader_args(self, ent=0.5, rel=0.5, numworkers=1, batch_size=1024, ):
        self.ent_neg_rate = ent
        self.rel_neg_rate = rel
        self.numworkers = numworkers
        self.batch_size = batch_size
        self.init()

    def set_pre_model(self, path):
        if os.path.exists(path):
            self.premodel = path
            self.loadmod = True
        else:
            print('[ERR] The pretrained model path is valid. The path is %s.' % path)

    def set_pre_emb(self, loademb, entfile, relfile):
        if os.path.exists(entfile) and os.path.exists(relfile):
            self.entityfile = entfile
            self.relationfile = relfile
            if loademb:
                self.loadembed = True


class TransE_config():
    def __init__(self):
        self.margin = 2.0
        self.hidden_size = 100
        self.L = 2
        self.regularization = 0.5

        self.entTotal = 0
        self.relTotal = 0
        self.batch_size = 0
        self.batch_seq_size = int(self.batch_size * 2)


class TransD_config():
    def __init__(self):
        self.margin = 1.0
        self.hidden_size = 50
        self.L = 2
        self.regularization = 0.25

        self.entTotal = 0
        self.relTotal = 0
        self.batch_size = 0
        self.batch_seq_size = int(self.batch_size * 2)


class TransH_config():
    def __init__(self):
        self.margin = 1.0
        self.hidden_size = 100
        self.L = 2
        self.C = 0.01
        self.eps = 0.001

        self.entTotal = 0
        self.relTotal = 0
        self.batch_size = 0
        self.batch_seq_size = int(self.batch_size * 2)


class TransA_config():
    def __init__(self):
        self.margin = 3.2
        self.hidden_size = 200
        self.L = 2
        self.lamb = 0.01
        self.regularization = 0.2

        self.entTotal = 0
        self.relTotal = 0
        self.batch_size = 1024
        self.batch_seq_size = int(self.batch_size * 2)

class ConvE_config():
    def __init__(self):
        self.hidden_size = 200
        self.label_smoothing_epsilon = 0.1
        self.use_bias = True

        self.input_dropout = 0.2
        self.feature_map_dropout = 0.2
        self.dropout = 0.3

class RotatE_config():
    def __init__(self):
        self.negative_adversarial_sampling = True
        self.adversarial_temperature = 1.0
        self.regularization = 0
        self.uni_weight = True
        self.gamma = 12.0
        self.hidden_size = 256 # model will automatically double it

class SimplE_config():
    def __init__(self):
        self.regularization =0.3
        self.L = 2
        self.hidden_size = 200


