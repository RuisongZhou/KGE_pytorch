import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import Config as Config
import models.RotatE as RotatE
from examples.base import TrainBase, adjust_learning_rate
import torch.nn.functional as F
import utils.util as util
import torch
from dataloader.dataloader import *


class RotatEModel(TrainBase):
    def __init__(self, config):
        super(RotatEModel, self).__init__(config)
        self.args = config
        self.model = RotatE.RotatE(self.args.modelparam)
        print(self.model)


    def get_iterator(self):
        train_dataloader_head = DataLoader(
            AdversarialDataset(self.args.trainpath, self.args.modelparam.entTotal, 'head-batch'),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.numworkers,
            drop_last=self.args.drop_last,
            collate_fn=AdversarialDataset.collate_fn)

        train_dataloader_tail = DataLoader(
            AdversarialDataset(self.args.trainpath, self.args.modelparam.entTotal, 'tail-batch'),
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.numworkers,
            drop_last=self.args.drop_last,
            collate_fn=AdversarialDataset.collate_fn)

        test_dataloader = DataLoader(
            TestDataset(self.args.testpath),
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.evalnumberworkers,
            shuffle=False,
            drop_last=False
        )

        return [train_dataloader_head, train_dataloader_tail], test_dataloader

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

        train_iterator, test_dataloader = self.get_iterator()
        for epoch in range(epochs):
            globalepoch += 1
            print("=" * 20 + "EPOCHS(%d/%d)" % (globalepoch, epochs) + "=" * 20)
            step = 0
            model.train()
            for dataloader in train_iterator:
                for pos_sample, neg_sample, subsampling_weight, mode in dataloader:
                    if self.args.usegpu:
                        pos_sample = pos_sample.cuda()
                        neg_sample = neg_sample.cuda()
                        subsampling_weight = subsampling_weight.cuda()

                    negative_score = model(neg_sample, mode=mode[0])
                    positive_score = model(pos_sample, mode=mode[0])

                    loss, positive_sample_loss, negative_sample_loss = model.loss(positive_score, negative_score,
                                                                                  subsampling_weight)

                    if self.args.usegpu:
                        lossVal = torch.sum(loss).cpu().item()
                        pl = torch.sum(positive_sample_loss).cpu().item()
                        nl = torch.sum(negative_sample_loss).cpu().item()
                    else:
                        lossVal = torch.sum(loss).item()
                        pl = torch.sum(positive_sample_loss).item()
                        nl = torch.sum(negative_sample_loss).item()

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
                    self.sumwriter.add_scalar('RotatE/train/loss', lossVal, global_step=globalstep)
                    self.sumwriter.add_scalar('RotatE/train/positive_sample_loss', pl, global_step=globalstep)
                    self.sumwriter.add_scalar('RotatE/train/negative_sample_loss', nl, global_step=globalstep)
                    self.sumwriter.add_scalar('RotatE/train/lr', lr, global_step=globalstep)

            if globalepoch % self.args.lrdecayepoch == 0:
                adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                lr = lr * self.args.lrdecay

            if globalepoch % self.args.evalepoch == 0:
                # eval the model
                print('begin eval the model')
                model.eval()
                for mode in ['head-batch', 'tail-batch']:
                    evalstep = 0
                    hit10_ = 0
                    hit1_ = 0
                    mr = 0
                    for data in test_dataloader:
                        evalstep += 1
                        if self.args.usegpu:
                            data = data.cuda()

                        ranks, hit1, hit10 = model.eval_model(data, mode=mode)
                        if evalstep % 1000 == 0:
                            print("[TEST-EPOCH(%d/%d)-STEP(%d)]mr:%f, hit@10:%f" % (
                                globalepoch, epochs, evalstep, ranks, hit10))

                        mr += ranks
                        hit10_ += hit10
                        hit1_ += hit1

                    mr /= evalstep
                    hit10_ /= evalstep
                    hit1_ /= evalstep
                    title = 'RotatE/test/' + mode
                    self.sumwriter.add_scalar(title+'/hit@10', hit10_, global_step=epoch + 1)
                    self.sumwriter.add_scalar(title+'/hit@1', hit1_, global_step=epoch + 1)
                    self.sumwriter.add_scalar(title+'/MR', mr, global_step=epoch + 1)
                variable_list = {
                    'step': globalstep,
                    'lr': lr,
                    'MR': mr,
                    'hit@10': hit10_
                }
                self.save_model(model, optimizer, variable_list)


if __name__ == '__main__':
    config = Config.Config()
    config.model = 'RotatE'
    config.batch_size = 256
    config.eval_batch_size = 16
    config.learningrate = 0.0001
    config.lrdecay = 1.0
    config.init()
    util.printArgs(config)
    util.printArgs(config.modelparam)
    model = RotatEModel(config)

    if config.init_checkpoint:
        optimizer = model.load_opt(model.model)
        model.load_model(model.model, optimizer)
        model.fit(model.model, optimizer)
    else:
        model.train()
