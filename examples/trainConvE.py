import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from examples.base import TrainBase, adjust_learning_rate
import Config as Config
import models.ConvE as ConvE
import utils.util as util
import torch
from torch.autograd import Variable

class ConvEModel(TrainBase):
    def __init__(self,args):
        super(ConvEModel, self).__init__(args)
        self.args = args
        self.model = ConvE.ConvE(args.modelparam)
        print(self.model)


    def load_model(self):
        print("INFO : Loading pre-training model!")
        modelType = os.path.splitext(self.args.premodel)[-1]
        if modelType == ".param":
            self.model.load_state_dict(torch.load(self.args.premodel))
        elif modelType == ".model":
            self.model = torch.load(self.args.premodel)
        else:
            print("ERROR : Model type %s is not supported!" % self.args.premodel)
            exit(1)


    def train(self):
        model = self.model
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
            for posData, _ in dataloader:

                if self.args.usegpu:
                    posData = Variable(torch.LongTensor(posData).cuda())
                    #negData = Variable(torch.LongTensor(negData).cuda())
                else:
                    posData = Variable(torch.LongTensor(posData))
                    #negData = Variable(torch.LongTensor(negData))

                # Calculate the loss from the modellrdecayepoch
                #data_batch = torch.cat((posData, negData), 0)

                loss = model(posData)
                if self.args.usegpu:
                    lossVal = torch.sum(torch.abs(loss)).cpu().item()
                else:
                    lossVal = torch.sum(torch.abs(loss)).item()

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
                hit10_ = 0
                hit1_ = 0
                mr = 0
                evalstep = 0
                for data in evaldataloader:
                    evalstep += 1
                    if self.args.usegpu:
                        data = Variable(torch.LongTensor(data).cuda())
                    else:
                        data = Variable(torch.LongTensor(data))

                    rank, hit10, hit1 = model.eval_model(data)
                    if evalstep % 1000 == 0:
                        print("[TEST-EPOCH(%d/%d)-STEP(%d)]mr:%f, hit@10:%f" % (
                            globalepoch, epochs, evalstep, rank, hit10))

                    mr +=rank
                    hit10_ +=hit10
                    hit1_ +=hit1

                mr /= evalstep
                hit10_ /= evalstep
                hit1_ /= evalstep
                self.sumwriter.add_scalar('ConvE/eval/hit@10', hit10_, global_step=epoch + 1)
                self.sumwriter.add_scalar('ConvE/eval/hit@1', hit1_, global_step=epoch + 1)
                self.sumwriter.add_scalar('ConvE/eval/MR', mr, global_step=epoch + 1)
                self.save_model(mr, hit10_, model)

    def eval_one_sample(self):
        pass



if __name__ == '__main__':
    config = Config.Config()
    config.model = 'ConvE'
    config.batch_size = 256
    config.eval_batch_size = 128
    config.learningrate = 0.003
    config.lrdecay = 1.0
    config.init()
    util.printArgs(config)

    model = ConvEModel(config)
    model.train()
