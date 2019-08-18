# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-11 11:15:04
@LastEditTime: 2019-08-18 16:49:32
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
from metrics import *
from models import *
from trainer import *

def main_pca(used_labels=None, subdir='PCA'):
    from sklearn.decomposition import PCA
    
    validset = MNIST('valid', used_labels)
    pca = PCA(n_components=3)
    mat = pca.fit_transform(validset.images.reshape(validset.images.shape[0], -1))
    
    logdir = os.path.join(configer.logdir, subdir) if subdir is not None else configer.logdir
    with SummaryWriter(logdir) as w:
        w.add_embedding(mat, validset.labels)

def main_crossent(feature_size, used_labels=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = Network(trainset.n_classes, feature_size=feature_size)
    params = net.parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = SupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True)
    trainer.train()
    del trainer

# ==============================================================================================================================
def main_unsupervised_entropy(feature_size, n_clusters=50, lamb=1.0, entropy_type='shannon', lr_m=1.0, used_labels=None, show_embedding=True, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkUnsupervised(feature_size)
    criterion = LossUnsupervisedEntropy(n_clusters, feature_size, lamb, entropy_type)
    params = [
        {'params': net.parameters(), }, 
        {'params': criterion.parameters(), 'lr': lr_m * configer.lrbase}
    ]
    
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_unsupervised_weighted_sum(feature_size, n_clusters=50, batchnorm=False, lamb=1.0, entropy_type='shannon', lr_m=1.0, used_labels=None, show_embedding=True, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkUnsupervised(feature_size, batchnorm=batchnorm)
    criterion = LossUnsupervisedWeightedSum(n_clusters, feature_size, lamb, entropy_type)
    params = [
        {'params': net.parameters(), }, 
        {'params': criterion.parameters(), 'lr': lr_m * configer.lrbase}
    ]
    
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_unsupervised_weighted_sum_with_encoder_decoder(feature_size, n_clusters=50, lamb=1.0, entropy_type='shannon', lr_m=1.0, used_labels=None, show_embedding=True, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkUnsupervisedWithEncoderDecoder(feature_size)
    criterion = LossUnsupervisedWithEncoderDecoder(n_clusters, feature_size, lamb, entropy_type)
    params = [
        {'params': net.parameters(), }, 
        {'params': criterion.parameters(), 'lr': lr_m * configer.lrbase}
    ]
    
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainerWithEncoderDecoder(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_unsupervised_sigma_i(feature_size, n_clusters=50, batchnorm=False, lamb=1.0, entropy_type='shannon', lr_m=1.0, used_labels=None, show_embedding=True, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkUnsupervised(feature_size, batchnorm=batchnorm)
    criterion = LossUnsupervisedSigmaI(n_clusters, feature_size, lamb, entropy_type)
    params = [
        {'params': net.parameters(), }, 
        {'params': criterion.parameters(), 'lr': lr_m * configer.lrbase}
    ]
    
    optimizer = optim.SGD
    lr_scheduler = MultiStepLR

    trainer = UnsupervisedTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

if __name__ == "__main__":

    # main_unsupervised_weighted_sum(3, 50, batchnorm=True, entropy_type='shannon', 
    #                     subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[baseline_bn]'.\
    #                                     format('shannon', 50, 3))
    # main_unsupervised_weighted_sum(3, 50, batchnorm=False, entropy_type='shannon', 
    #                     subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[baseline]'.\
    #                                     format('shannon', 50, 3))

    # main_unsupervised_weighted_sum_with_encoder_decoder(3, 50, entropy_type='shannon', 
    #                     subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[baseline_with_encoder_decoder]'.\
    #                                     format('shannon', 50, 3))

    main_unsupervised_sigma_i(3, 50, batchnorm=False, 
                        subdir='unsupervised_sigma_i_c{:3d}_f{:3d}_[baseline]'.\
                                        format(50, 3))

    # ===========================================================================================================
    ## shannon
    ### 选择lambda
    # for lamb in [5**i for i in range(6)]:    # 1, 5, 25, 125, 625, 3125
    #     main_unsupervised_entropy(3, 50, lamb=lamb, entropy_type='shannon', 
    #                         subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[lamb]{:4d}'.\
    #                                         format('shannon', 50, 3, lamb))

    ### 选择聚类数目
    # for num_clusters in [20 * (i + 1) for i in range(5)]:    # 20, 40, 60, 80, 100
    #     main_unsupervised_entropy(3, num_clusters, lamb=TODO, entropy_type='shannon', 
    #                         subdir='unsupervised_{:s}_[c]{:3d}_f{:3d}_lamb{:4d}'.\
    #                                         format('shannon', num_clusters, 3, TODO))

    ### 选择学习率倍数
    # for lr_m in [3**i for i in range(6)]:    # 1, 3, 9, 27, 81, 243
    #     main_unsupervised_entropy(3, 50, lamb=TODO, entropy_type='shannon', lr_m=lr_m
    #                         subdir='unsupervised_{:s}_c{:3d}_f{:3d}_lamb{:4d}_[lrm]{:4d}'.\
    #                                         format('shannon', 50, 3, TODO, lr_m))
