import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tensorboardX import SummaryWriter
from dataloader.dataloader import *
from torch.utils.data import DataLoader


def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class TrainBase():
    def __init__(self, args):
        self.args = args
        self.sumwriter = SummaryWriter(log_dir=args.summarydir)

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
                                batch_size=self.args.eval_batch_size,
                                shuffle=False,
                                num_workers=self.args.evalnumberworkers,
                                drop_last=False)

        return dataloader

    def load_model(self):
        pass

    def save_model(self, mr, hit10, model):
        name = self.args.model
        path = os.path.join(self.args.modelpath, "%s_mr_%d__hit10_%f.model" % (name, mr, hit10))
        torch.save(model, path)

    def load_opt(self, model):
        lr = self.args.learningrate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        elif OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        elif OPTIMIZER == 'Adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        elif OPTIMIZER == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=self.args.weight_decay, lr=lr)
        else:
            print("ERROR : Optimizer %s is not supported." % OPTIMIZER)
            print("Support optimizer:\n===>Adam\n===>SGD\n===>Adagrad\n===>Adadelta")
            exit(1)
        return optimizer

    def fit(self, model):
        epochs = self.args.epochs
        lr = self.args.learningrate
        optimizer = self.load_opt(model)

        if self.args.usegpu:
            model.cuda()
        globalstep = 0
        globalepoch = 0
        minLoss = float("inf")

        dataloader = self.load_data()
        evaldataloader = self.load_data(mode='valid')

        for epoch in range(epochs):
            globalepoch += 1
            print("=" * 20 + "EPOCHS(%d/%d)" % (globalepoch, epochs) + "=" * 20)
            step = 0
            model.train()
            for posData, negData in dataloader:

                # Normalize the embedding if neccessary
                model.normalizeEmbedding()

                if self.args.usegpu:
                    posData = Variable(torch.LongTensor(posData).cuda())
                    negData = Variable(torch.LongTensor(negData).cuda())
                else:
                    posData = Variable(torch.LongTensor(posData))
                    negData = Variable(torch.LongTensor(negData))

                # Calculate the loss from the modellrdecayepoch
                data_batch = torch.cat((posData, negData), 0)

                loss = model(data_batch)
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
                if step % 50 == 0:
                    print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%f, minLoss:%f" % (
                        epoch + 1, epochs, step, lossVal, minLoss))
                step += 1
                globalstep += 1
                self.sumwriter.add_scalar('train/loss', lossVal, global_step=globalstep)
                self.sumwriter.add_scalar('train/lr', lr, global_step=globalstep)

            if globalepoch % self.args.lrdecayepoch == 0:
                adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                lr = lr * self.args.lrdecay

            if globalepoch % self.args.evalepoch == 0:
                # eval the model
                print('begin eval the model')
                model.eval()
                hit10 = 0
                mr = 0
                evalstep = 0
                for data in evaldataloader:
                    evalstep += 1
                    if self.args.usegpu:
                        data = Variable(torch.LongTensor(data).cuda())
                    else:
                        data = Variable(torch.LongTensor(data))

                    rankH, rankT = model.eval_model(data)
                    if evalstep % 1000 == 0:
                        print("[TEST-EPOCH(%d/%d)-STEP(%d)]mr:%f, hit@10:%f" % (
                        globalepoch, epochs, evalstep, mr/evalstep, hit10/evalstep))
                    mr += (rankH + rankT) / 2
                    if rankT <= 10:
                        hit10 += 1

                mr /= evalstep
                hit10 /= evalstep
                self.sumwriter.add_scalar('eval/hit@10', hit10, global_step=epoch + 1)
                self.sumwriter.add_scalar('eval/MR', mr, global_step=epoch + 1)
                self.save_model(mr, hit10, model)
