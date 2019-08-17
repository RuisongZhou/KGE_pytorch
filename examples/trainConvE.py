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
    def __init__(self, args):
        super(ConvEModel, self).__init__(args)
        self.args = args
        self.model = ConvE.ConvE(args.modelparam)
        print(self.model)

    def train(self, optimizer=None):
        model = self.model
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
            for posData, _ in train_iterator:
                posData = posData.reshape(-1,3)
                if self.args.usegpu:
                    posData = posData.cuda()
                    # negData = Variable(torch.LongTensor(negData).cuda())

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
                self.sumwriter.add_scalar('ConvE/train/loss', lossVal, global_step=globalstep)
                self.sumwriter.add_scalar('ConvE/train/lr', lr, global_step=globalstep)

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
                for data in test_iterator:
                    evalstep += 1
                    if self.args.usegpu:
                        data = data.cuda()

                    rank, hit10, hit1 = model.eval_model(data)
                    if evalstep % 10 == 0:
                        print("[TEST-EPOCH(%d/%d)-STEP(%d)]mr:%f, hit@10:%f" % (
                            globalepoch, epochs, evalstep, rank, hit10))

                    mr += rank
                    hit10_ += hit10
                    hit1_ += hit1

                mr /= evalstep
                hit10_ /= evalstep
                hit1_ /= evalstep
                self.sumwriter.add_scalar('ConvE/eval/hit@10', hit10_, global_step=epoch + 1)
                self.sumwriter.add_scalar('ConvE/eval/hit@1', hit1_, global_step=epoch + 1)
                self.sumwriter.add_scalar('ConvE/eval/MR', mr, global_step=epoch + 1)
                variable_list = {
                    'step': globalstep,
                    'lr': lr,
                    'MR': mr,
                    'hit@10': hit10_
                }
                self.save_model(model, optimizer, variable_list)

        print("=" * 20 + "FINISH TRAINING" + "=" * 20)
        self.valid(model, valid_iterator)


if __name__ == '__main__':
    config = Config.Config()
    config.model = 'ConvE'
    config.batch_size = 256
    config.eval_batch_size = 128
    config.learningrate = 0.003
    config.lrdecay = 1.0
    config.init()
    util.printArgs(config)
    util.printArgs(config.modelparam)

    model = ConvEModel(config)
    optimizer = model.load_opt(model.model)
    if config.init_checkpoint:
        model.load_model(model.model, optimizer)
        model.train(optimizer)
    else:
        model.train()
