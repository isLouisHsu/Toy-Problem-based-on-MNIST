import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import MNIST
from metrics import LossUnsupervised, MarginLoss, MarginLossWithParameter
from models import Network, NetworkMargin
from trainer import SupervisedTrainer, UnsupervisedTrainer, MarginTrainer, MarginTrainerWithParameter

def main_crossent(num_classes, feature_size):
    net = Network(num_classes=num_classes, feature_size=feature_size)
    params = net.parameters()
    trainset = MNIST('train')
    validset = MNIST('valid')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = SupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()
    del trainer

def main_margin(num_classes=10, feature_size=2, s=8.0, m1=2.00, m2=0.5, m3=0.35, m4=0.5, subdir=None):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.cosine_layer.weights), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.cosine_layer.weights, 'weight_decay': 4e-4},
    ]

    trainset = MNIST('train'); validset = MNIST('valid')
    criterion = MarginLoss(s, m1, m2, m3, m4)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_adaptivemargin(num_classes=10, feature_size=2, subdir=None):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)
    criterion = MarginLossWithParameter(num_classes)

    base_params = list(filter(lambda x: id(x) != id(net.cosine_layer.weights), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.cosine_layer.weights, 'weight_decay': 4e-4},
        {'params': criterion.parameters(), 'weight_decay': 4e-4},
    ]

    trainset = MNIST('train'); validset = MNIST('valid')
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainerWithParameter(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

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
    del trainer

if __name__ == "__main__":

    # modified
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_feat2')
    # cosface
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_feat2')
    # sphereface
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_feat2')
    # arcface
    main_margin(num_classes=10, feature_size=2, s=16.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_feat2_s16')
    main_margin(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_feat2')
    main_margin(num_classes=10, feature_size=2, s= 4.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_feat2_s4')
    main_margin(num_classes=10, feature_size=2, s= 1.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_feat2_s1')

    # modified
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_feat3')
    # cosface
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_feat3')
    # sphereface
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_feat3')
    # arcface
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_feat3')

if __name__ == "__main__":

    # cosmulface
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.50, subdir='cosmulface_feat2_m41.50')
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.25, subdir='cosmulface_feat2_m41.25')
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.00, subdir='cosmulface_feat2_m41.00')
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=0.75, subdir='cosmulface_feat2_m40.75')
    main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=0.50, subdir='cosmulface_feat2_m40.50')
    # adaptiveface
    main_adaptivemargin(num_classes=10, feature_size=2, subdir='adaptiveface_feat2')

    # cosmulface
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.50, subdir='cosmulface_feat3_m41.50')
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.25, subdir='cosmulface_feat3_m41.25')
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.00, subdir='cosmulface_feat3_m41.00')
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=0.75, subdir='cosmulface_feat3_m40.75')
    main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=0.50, subdir='cosmulface_feat3_m40.50')
    # adaptiveface
    main_adaptivemargin(num_classes=10, feature_size=3, subdir='adaptiveface_feat3')

    exit(0)
