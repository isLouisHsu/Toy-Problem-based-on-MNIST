import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import MNIST
from metrics import UnsupervisedLoss
from models import Net
from trainer import Trainer

def main_supervised():
    net = Net(outp=10)
    params = net.parameters()
    trainset = MNIST('train')
    validset = MNIST('valid')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = Trainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1)
    trainer.train()

def main_unsupervised():
    net = net(outp=2)
    

if __name__ == "__main__":
    main_supervised()