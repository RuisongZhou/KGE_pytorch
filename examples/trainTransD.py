import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from examples.base import TrainBase
import Config as Config
import models.TransD as TransD
import utils.util as util
import torch


class TransDModel(TrainBase):
    def __init__(self, args):
        super(TransDModel, self).__init__(args)
        self.args = args
        self.model = TransD.TransD(args.modelparam)
        print(self.model)

    def train(self):
        model.fit(self.model)


if __name__ == '__main__':
    config = Config.Config()
    config.model = 'TransD'
    config.batch_size = 200
    config.numworkers = 12
    config.evalnumberworkers = 16
    config.learningrate = 0.005
    config.lrdecay = 0.98
    config.init()
    config.modelparam.margin = 3.0

    util.printArgs(config)
    util.printArgs(config.modelparam)

    model = TransDModel(config)
    optimizer = model.load_opt(model.model)
    if config.init_checkpoint:
        model.load_model(model.model, optimizer)
        model.fit(model.model, optimizer)
    else:
        model.train()
