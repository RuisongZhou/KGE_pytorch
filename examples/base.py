import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tensorboardX import SummaryWriter
from dataloader.dataloader import *
from torch.utils.data import DataLoader
import json


def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class TrainBase():
    def __init__(self, args):
        self.args = args
        self.sumwriter = SummaryWriter(log_dir=args.summarydir)

    def get_iterator(self):
        train_iterator = DataLoader(
            UniformDataSet(self.args.trainpath, self.args.neg_sample_rate),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.numworkers,
            drop_last=self.args.drop_last
        )
        test_iterator = DataLoader(
            TestDataset(self.args.testpath),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.evalnumberworkers,
            drop_last=False
        )
        valid_iterator = DataLoader(
            TestDataset(self.args.validpath),
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.evalnumberworkers,
            drop_last=False
        )

        return train_iterator, test_iterator, valid_iterator

    def load_model(self, model, optimizer):
        print('Loading checkpoint %s...' % self.args.init_checkpoint)
        checkpoint = torch.load(self.args.init_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        current_learning_rate = checkpoint['lr']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model(self, model, optimizer, variable_list):

        torch.save({
            **variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(self.args.savepath, 'checkpoint'))

    #【TODO】could be save with timestamp
    def save_model_param(self, model):
        param = model.getWeight()
        filename = os.path.join(self.args.savepath, 'checkpoint')
        with open(filename + '.json', 'a') as outfile:
            json.dump(param, outfile, ensure_ascii=False)
            outfile.write('\n')

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
            raise EnvironmentError
        return optimizer

    def fit(self, model, optimizer=None):
        epochs = self.args.epochs
        lr = self.args.learningrate
        if not optimizer:
            optimizer = self.load_opt(model)
        if self.args.usegpu:
            model.cuda()

        globalstep = 0
        globalepoch = 0
        minLoss = float("inf")

        train_iterator, test_iterator, valid_iterator = self.get_iterator()

        for epoch in range(epochs):
            globalepoch += 1
            print("=" * 20 + "EPOCHS(%d/%d)" % (globalepoch, epochs) + "=" * 20)
            step = 0
            model.train()
            for posData, negData in train_iterator:
                posData = posData.reshape(-1,3)
                negData = negData.reshape(-1,3)
                if self.args.usegpu:
                    posData = posData.cuda()
                    negData = negData.cuda()

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
                self.sumwriter.add_scalar(self.args.model + '/train/loss', lossVal, global_step=globalstep)
                self.sumwriter.add_scalar(self.args.model + '/train/lr', lr, global_step=globalstep)

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
                mr_t = 0
                mr_h = 0
                for data in test_iterator:
                    evalstep += 1
                    if self.args.usegpu:
                        data = data.cuda()

                    rankH, rankT = model.eval_model(data)
                    mr_h +=rankH
                    mr_t +=rankT
                    if evalstep % 5000 == 0:
                        print("[TEST-EPOCH(%d/%d)-STEP(%d)]mr:%f, hit@10:%f" % (
                            globalepoch, epochs, evalstep, mr / evalstep, hit10 / evalstep))
                    mr += (rankH + rankT) / 2
                    if (rankH + rankT) / 2 <= 10:
                        hit10 += 1

                mr /= evalstep
                hit10 /= evalstep
                mr_t /=evalstep
                mr_h /=evalstep
                self.sumwriter.add_scalar(self.args.model + '/eval/hit@10', hit10, global_step=epoch + 1)
                self.sumwriter.add_scalar(self.args.model + '/eval/MR', mr, global_step=epoch + 1)
                self.sumwriter.add_scalar(self.args.model + '/eval/RankT', mr_t, global_step=epoch + 1)
                self.sumwriter.add_scalar(self.args.model + '/eval/RankH', mr_h, global_step=epoch + 1)
                variable_list = {
                    'step': globalstep,
                    'lr': lr,
                    'MR': mr,
                    'hit@10': hit10
                }
                self.save_model(model, optimizer, variable_list)

        print("=" * 20 + "FINISH TRAINING" + "=" * 20)
        self.valid(model, valid_iterator)

    def valid(self, model, iter):
        print('begin eval the model')
        model.eval()
        hit10 = 0
        mr = 0
        evalstep = 0
        for data in iter:
            evalstep += 1
            if self.args.usegpu:
                data = data.cuda()

            rankH, rankT = model.eval_model(data)
            if evalstep % 1000 == 0:
                print("[VALID-STEP(%d)]mr:%f, hit@10:%f" % (
                    evalstep, mr / evalstep, hit10 / evalstep))
            mr += (rankH + rankT) / 2
            if rankT <= 10:
                hit10 += 1

        mr /= evalstep
        hit10 /= evalstep
        print("=" * 20 + "VALID RESULTS" + "=" * 20)
        print('Mean Rank: %f' % mr)
        print('Hit@10: %d' % hit10)
