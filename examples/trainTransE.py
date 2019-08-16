import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataloader.dataloader import *
from torch.utils.data import DataLoader
import Config as Config
import models.TransE as TransE
import utils.util as util
import utils.evaluate as evaluate

def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class TrainModel():
    def __init__(self, args):
        self.args = args
        self.model = TransE.TransE(args.modelparam)
        print(self.model)

    def load_data(self, mode='train'):
        if mode == 'train':
            dataset = EmDataSet(self.args.trainpath)
            dataset.generate_neg_samples()
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.batch_size,
                                    shuffle=self.args.shuffle,
                                    num_workers=self.args.numworkers,
                                    drop_last=self.args.drop_last)
            return dataloader
        elif mode == 'test':
            dataset = EmDataSet(self.args.testpath, mode='test')
        elif mode == 'valid':
            dataset = EmDataSet(self.args.validpath, mode='eval')
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=self.args.evalnumberworkers,
                                drop_last=False)

        return dataloader

    def load_eval_data(self):
        _, tripleDict = util.loadTriple(self.args.trainpath)
        dataset = EmDataSet(self.args.validpath, mode='eval')
        dataloader = DataLoader(dataset,
                                batch_size= self.args.batch_seq_size,
                                shuffle=False,
                                drop_last=False)
        return tripleDict, dataloader

    def load_embedding(self):
        print("INFO : Loading pre-training entity and relation embedding!")
        self.model.load_emb_weight(entityEmbedFile=self.args.entityfile,
                                   relationEmbedFile=self.args.relationfile)

    def load_model(self):
        print("INFO : Loading pre-training TransE model!")
        modelType = os.path.splitext(self.args.premodel)[-1]
        if modelType == ".param":
            self.model.load_state_dict(torch.load(self.args.premodel))
        elif modelType == ".model":
            self.model = torch.load(self.args.premodel)
        else:
            print("ERROR : Model type %s is not supported!")
            exit(1)

    def save_model(self, mr,hit):
        path = os.path.join(self.args.modelpath, "TransE_mr_%d__hit10_%f.model" % (mr, hit))
        torch.save(self.model, path)

    def fit(self):
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        lr = self.args.learningrate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),weight_decay = self.args.weight_decay, lr=lr)
        elif OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), weight_decay=self.args.weight_decay,lr=lr)
        elif OPTIMIZER == 'Adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        elif OPTIMIZER == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        else:
            print("ERROR : Optimizer %s is not supported." % OPTIMIZER)
            print("Support optimizer:\n===>Adam\n===>SGD\n===>Adagrad\n===>Adadelta")
            exit(1)

        if self.args.usegpu:
            self.model.cuda()
        globalstep = 0
        globalepoch =0
        minLoss = float("inf")

        dataloader = self.load_data()
        evaldataloader = self.load_data(mode='valid')

        for i in range(epochs):
            globalepoch+=1
            print("=" * 20 + "EPOCHS(%d/%d)" % (globalepoch, epochs) + "=" * 20)
            step = 0
            self.model.train()
            for posData, negData in dataloader:

                # Normalize the embedding if neccessary
                self.model.normalizeEmbedding()

                if self.args.usegpu:
                    posData = Variable(torch.LongTensor(posData).cuda())
                    negData = Variable(torch.LongTensor(negData).cuda())
                else:
                    posData = Variable(torch.LongTensor(posData))
                    negData = Variable(torch.LongTensor(negData))

                # Calculate the loss from the modellrdecayepoch
                data_batch = torch.cat((posData, negData), 0)

                loss = self.model(data_batch)
                if self.args.usegpu:
                    lossVal = loss.cpu().item()
                else:
                    lossVal = loss.item()

                # Calculate the gradient and step down
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print infomation and add to summary
                if minLoss > lossVal:
                    minLoss = lossVal
                print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%f, minLoss:%f" % (
                    i + 1, epochs, step, lossVal, minLoss))
                step += 1
                globalstep += 1
                sumwriter.add_scalar('train/loss', lossVal, global_step=globalstep)
                sumwriter.add_scalar('train/lr', lr, global_step=globalstep)

            if globalepoch % self.args.lrdecayepoch == 0:
                adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                lr = lr * self.args.lrdecay

            if globalepoch % self.args.evalepoch == 0:
                # eval the model
                print('begin eval the model')
                hit10 = 0
                mr = 0
                num = 0
                for data in evaldataloader:
                    num+=1
                    if self.args.usegpu:
                        data = Variable(torch.LongTensor(data).cuda())
                    else:
                        data = Variable(torch.LongTensor(data))

                    rankH, rankT = self.model.eval_model(data)
                    if num%1000 == 0:
                        print("[TEST-EPOCH(%d/%d)-STEP(%d)]rankH:%d, rankT:%d" % (globalepoch, epochs, num, rankH, rankT))
                    mr += (rankH+rankT)/2
                    if rankT <= 10:
                        hit10+=1

                mr/=num
                hit10/=num
                sumwriter.add_scalar('eval/hit@10', hit10, global_step=i + 1)
                sumwriter.add_scalar('eval/MR', mr, global_step=i + 1)
                self.save_model(mr, hit10)




if __name__ == '__main__':
    config = Config.Config()
    config.model = 'TransE'
    config.init()
    util.printArgs(config)
    sumwriter = SummaryWriter(log_dir=config.summarydir)

    model = TrainModel(config)
    if config.loadmod:
        model.load_model()

    model.fit()
