import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import MNIST
from metrics import LossUnsupervised
from models import NetSupervised, NetUnsupervised
from trainer import SupervisedTrainer, UnsupervisedTrainer

def main_supervised():
    net = NetSupervised(num_classes=10)
    params = net.parameters()
    trainset = MNIST('train')
    validset = MNIST('valid')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = SupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1)
    trainer.train()

def main_unsupervised(num_clusters, feature_size):
    net = NetUnsupervised(num_clusters, feature_size)
    criterion = LossUnsupervised(num_clusters, feature_size)
    params = [{'params': net.parameters(), }, {'params': criterion.m, }]
    trainset = MNIST('train')
    validset = MNIST('valid')
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1)
    trainer.train()
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--unsupervised', '-u', action='store_true')
    parser.add_argument('--num_clusters', '-n', type=int, default=10)
    parser.add_argument('--feature_size', '-f', type=int, default=128)
    args = parser.parse_args()

    if args.unsupervised:
        main_unsupervised(args.num_clusters, args.feature_size)
    
    else:
        main_supervised()
