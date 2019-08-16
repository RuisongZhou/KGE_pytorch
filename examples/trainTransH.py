import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from examples.base import TrainBase
import Config as Config
import models.TransH as TransH
import utils.util as util
import torch


class TransHModel(TrainBase):
    def __init__(self, args):
        super(TransHModel, self).__init__(args)
        self.args = args
        self.model = TransH.TransH(args.modelparam)
        print(self.model)

    def train(self):
        model.fit(self.model)


if __name__ == '__main__':
    config = Config.Config()
    config.model = 'TransH'
    config.init()
    util.printArgs(config)
    util.printArgs(config.modelparam)

    model = TransHModel(config)
    optimizer = model.load_opt(model.model)
    if config.init_checkpoint:
        model.load_model(model.model, optimizer)
        model.fit(model.model, optimizer)
    else:
        model.train()
