import torch
import random
import numpy as np
from datapro import Simdata_pro,loading_data
from DDQN_Algorithum1 import DDQN
from train import train_test
from datetime import datetime

class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch = 1
        self.gcn_layers = 3
        self.view = 3
        self.fm = 128
        self.fd = 128
        self.oi = 128
        self.inSize = 128
        self.outSize = 128
        self.nodeNum = 64
        self.hdnDropout = 0.5
        self.fcDropout = 0.5
        self.maskMDI = False
        self.alpha = 0.005
        self.device = torch.device('cpu')


def main():
    param = Config()
    demo_1 = DDQN()
    simData = Simdata_pro(param)
    train_dex, valid_dex, sample, true_label, train_data, allneg_samples = loading_data(param)
    demo_1.train(train_dex, valid_dex, sample, true_label, simData, train_data, param, allneg_samples)


if __name__ == "__main__":
    now = datetime.now()
    start = now.strftime
    main()
