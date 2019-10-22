# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-10-22 10:34:45
@LastEditTime: 2019-10-22 10:37:45
@Update: 
'''
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter

from config import configer
from datasets import MNIST
from metrics import LossSupervisedNew
from models import Network
from trainer import SupervisedTrainer

def main_supervised_new(used_labels=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = Network(trainset.n_classes, feature_size=128)
    params = net.parameters()
    criterion = LossSupervisedNew()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = SupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()
    del trainer

if __name__ == "__main__":
    
    main_supervised_new()