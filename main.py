import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import MNIST
from metrics import LossUnsupervised, MarginLoss
from models import Network, NetworkMargin
from trainer import SupervisedTrainer, UnsupervisedTrainer, MarginTrainer

def main_crossent(num_classes, feature_size):
    net = Network(num_classes=num_classes, feature_size=feature_size)
    params = net.parameters()
    trainset = MNIST('train')
    validset = MNIST('valid')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = SupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1)
    trainer.train()

def main_cosmargin(num_classes=10, feature_size=2):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.center), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.center, 'weight_decay': 4e-4},
    ]

    trainset = MNIST('train'); validset = MNIST('valid')
    criterion = MarginLoss(s=32.0, m1=0, m2=0, m3=0.35)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()

def main_spheremargin(num_classes=10, feature_size=2):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.center), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.center, 'weight_decay': 4e-4},
    ]

    trainset = MNIST('train'); validset = MNIST('valid')
    criterion = MarginLoss(s=32.0, m1=2.00, m2=0, m3=0)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()

def main_arcmargin(num_classes=10, feature_size=2):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.center), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.center, 'weight_decay': 4e-4},
    ]

    trainset = MNIST('train'); validset = MNIST('valid')
    criterion = MarginLoss(s=32.0, m1=0, m2=0.5, m3=0)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()

def main_unsupervised(num_classes, feature_size):
    net = Network(num_classes=num_classes, feature_size=feature_size)
    criterion = LossUnsupervised(num_classes, feature_size)
    # params = [{'params': net.parameters(), }, {'params': criterion.m, }]
    params = [{'params': net.parameters(), }, {'params': criterion.parameters(), }]
    trainset = MNIST('train')
    validset = MNIST('valid')
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1)
    trainer.train()
    trainer.show_embedding_features(validset)

if __name__ == "__main__":
    main_cosmargin()
    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--unsupervised', '-u', action='store_true')
    parser.add_argument('--losstype', '-lt', choices=['crossent', 'unsupervised', 'cosmargin', 'spheremargin', 'arcmargin'])
    parser.add_argument('--num_classes', '-n', type=int, default=10)
    parser.add_argument('--feature_size', '-f', type=int, default=2)
    args = parser.parse_args()

    if args.losstype == 'crossent':
        main_crossent(args.num_classes, args.feature_size)
    elif args.losstype == 'unsupervised':
        main_unsupervised(args.num_classes, args.feature_size)

    elif args.losstype == 'cosmargin':
        main_cosmargin()
    elif args.losstype == 'spheremargin':
        main_spheremargin()
    elif args.losstype == 'arcmargin':
        main_arcmargin()
