import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from examples.base import TrainBase
import Config as Config
import models.TransD as TransD
import utils.util as util
import torch

class TransDModel(TrainBase):
    def __init__(self,args):
        super(TransDModel, self).__init__(args)
        self.args = args
        self.model = TransD.TransD(args.modelparam)
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
    config.model = 'TransD'
    config.batch_size = 200
    config.numworkers = 12
    config.evalnumberworkers = 16
    config.learningrate = 0.005
    config.lrdecay = 0.98
    config.init()
    config.modelparam.margin = 3.0
    util.printArgs(config)

    model = TransDModel(config)
    model.train()
