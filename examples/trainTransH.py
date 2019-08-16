import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from examples.base import TrainBase
import Config as Config
import models.TransH as TransH
import utils.util as util
import torch

class TransHModel(TrainBase):
    def __init__(self,args):
        super(TransHModel, self).__init__(args)
        self.args = args
        self.model = TransH.TransH(args.modelparam)
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
        model.fit(self.model)

    def eval_one_sample(self):
        pass



if __name__ == '__main__':
    config = Config.Config()
    config.model = 'TransH'
    config.init()
    util.printArgs(config)

    model = TransHModel(config)
    model.train()
