# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-11 11:15:04
@LastEditTime: 2019-08-18 16:40:54
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
def main_margin(used_labels=None, feature_size=2, s=8.0, m1=2.00, m2=0.5, m3=0.35, m4=0.5, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkMargin(num_classes=trainset.n_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.cosine_layer.weights), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.cosine_layer.weights, 'weight_decay': 4e-4},
    ]

    criterion = MarginLoss(s, m1, m2, m3, m4)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainer(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_adaptivemargin(used_labels=None, feature_size=2, s=8.0, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkMargin(num_classes=trainset.n_classes, feature_size=feature_size)
    criterion = MarginLossWithParameter(trainset.n_classes, s)

    base_params = list(filter(lambda x: id(x) != id(net.cosine_layer.weights), net.parameters()))
    params = [
        {'params': base_params,              'weight_decay': 4e-5},
        {'params': net.cosine_layer.weights, 'weight_decay': 4e-4},
        {'params': criterion.parameters(),   'weight_decay': 4e-4},
    ]

    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainerWithParameter(configer, net, params, trainset, validset, criterion, 
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer

def main_margin_with_vector_loss(used_labels=None, feature_size=2, s=8.0, m1=2.00, m2=0.5, m3=0.35, m4=0.5, lda=0.2, subdir=None):
    trainset = MNIST('train', used_labels); validset = MNIST('valid', used_labels)
    net = NetworkMargin(num_classes=trainset.n_classes, feature_size=feature_size)

    base_params = list(filter(lambda x: id(x) != id(net.cosine_layer.weights), net.parameters()))
    params = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.cosine_layer.weights, 'weight_decay': 4e-4},
    ]

    criterion = MarginLoss(s, m1, m2, m3, m4)
    optimizer = optim.Adam
    lr_scheduler = MultiStepLR

    trainer = MarginTrainerWithVectorLoss(configer, net, params, trainset, validset, criterion, lda,
                    optimizer, lr_scheduler, num_to_keep=5, resume=False, valid_freq=1, show_embedding=True, subdir=subdir)
    trainer.train()
    del trainer


## ==============================================================================================================================
# 实验一：modified & sphereface & arcface & cosface
# if __name__ == "__main__":

#     # -------------------------------------------------------- dim=2 ---------------------------------------------------------
#     # modified
#     main_margin(used_labels=None, feature_size=2, s= 8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_dim2')
#     # sphereface
#     main_margin(used_labels=None, feature_size=2, s= 8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_dim2_m1=2.00')
#     # arcface
#     main_margin(used_labels=None, feature_size=2, s=16.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=16')
#     main_margin(used_labels=None, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=8')
#     main_margin(used_labels=None, feature_size=2, s= 4.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=4')
#     main_margin(used_labels=None, feature_size=2, s= 1.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=1')
#     # cosface
#     main_margin(used_labels=None, feature_size=2, s= 8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_dim2_m3=0.35')

#     # -------------------------------------------------------- dim=3 ---------------------------------------------------------
#     # modified
#     main_margin(used_labels=None, feature_size=3, s= 8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_dim3')
#     # sphereface
#     main_margin(used_labels=None, feature_size=3, s= 8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_dim3_m1=2.00')
#     # arcface
#     main_margin(used_labels=None, feature_size=3, s=16.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=16')
#     main_margin(used_labels=None, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=8')
#     main_margin(used_labels=None, feature_size=3, s= 4.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=4')
#     main_margin(used_labels=None, feature_size=3, s= 1.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=1')
#     # cosface
#     main_margin(used_labels=None, feature_size=3, s= 8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_dim3_m3=0.35')

#     exit(0)

# ==============================================================================================================================
# 实验二： adaptiveface
# if __name__ == "__main__":

#     # -------------------------------------------------------- dim=2 ---------------------------------------------------------
#     # adaptiveface
#     main_adaptivemargin(used_labels=None, feature_size=2, s=8.0, subdir='adaptiveface_dim2_F')

#     # -------------------------------------------------------- dim=3 ---------------------------------------------------------
#     # adaptiveface
#     main_adaptivemargin(used_labels=None, feature_size=3, s=8.0, subdir='adaptiveface_dim3_F')

#     exit(0)

# 实验二： cosmulface 
# if __name__ == "__main__":

#     # -------------------------------------------------------- dim=2 ---------------------------------------------------------
#     # cosmulface
#     main_margin(used_labels=None, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=2.00, subdir='cosmulface_dim2_m4=2.00')
#     main_margin(used_labels=None, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=3.00, subdir='cosmulface_dim2_m4=3.00')
#     main_margin(used_labels=None, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=4.00, subdir='cosmulface_dim2_m4=4.00')

#     # -------------------------------------------------------- dim=3 ---------------------------------------------------------
#     # cosmulface
#     main_margin(used_labels=None, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=2.00, subdir='cosmulface_dim3_m4=2.00')
#     main_margin(used_labels=None, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=3.00, subdir='cosmulface_dim3_m4=3.00')
#     main_margin(used_labels=None, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=4.00, subdir='cosmulface_dim3_m4=4.00')

#     exit(0)

# ==============================================================================================================================
# 实验三：arcface with_vector_loss
# if __name__ == "__main__":

#     # -------------------------------------------------------- dim=2 ---------------------------------------------------------
#     # arcface
#     main_margin_with_vector_loss(used_labels=None, feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=1.0, subdir='arcface_dim2_lda=1.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=2.0, subdir='arcface_dim2_lda=2.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=4.0, subdir='arcface_dim2_lda=4.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=8.0, subdir='arcface_dim2_lda=8.0')
    
#     # -------------------------------------------------------- dim=3 ---------------------------------------------------------
#     main_margin_with_vector_loss(used_labels=None, feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=1.0, subdir='arcface_dim3_lda=1.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=2.0, subdir='arcface_dim3_lda=2.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=4.0, subdir='arcface_dim3_lda=4.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=8.0, subdir='arcface_dim3_lda=8.0')
    
#     # -------------------------------------------------------- dim=64 --------------------------------------------------------
#     main_margin_with_vector_loss(used_labels=None, feature_size=64, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=1.0, subdir='arcface_dim64_lda=1.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size=64, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=2.0, subdir='arcface_dim64_lda=2.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size=64, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=4.0, subdir='arcface_dim64_lda=4.0')
#     main_margin_with_vector_loss(used_labels=None, feature_size=64, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=8.0, subdir='arcface_dim64_lda=8.0')

#     exit(0)

# 实验三：arcface with_vector_loss， 更少的类别
# if __name__ == "__main__":

#     # -------------------------------------------------------- dim=2 ---------------------------------------------------------
#     # arcface
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 0.0, subdir='arcface_dim2_lda=0.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 2.0, subdir='arcface_dim2_lda=2.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 8.0, subdir='arcface_dim2_lda=8.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=16.0, subdir='arcface_dim2_lda=16.0')
    
#     # -------------------------------------------------------- dim=3 ---------------------------------------------------------
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 0.0, subdir='arcface_dim3_lda=0.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 2.0, subdir='arcface_dim3_lda=2.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda= 8.0, subdir='arcface_dim3_lda=8.0')
#     main_margin_with_vector_loss(used_labels=[1, 2, 3, 4], feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=16.0, subdir='arcface_dim3_lda=16.0')
    
#     main_margin_with_vector_loss(used_labels=[1, 2, 3],       feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=16.0, subdir='arcface_dim3_lda=16.0_c3')   # n_classes = 3
#     main_margin_with_vector_loss(used_labels=[0, 1, 2, 3, 4], feature_size= 3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=16.0, subdir='arcface_dim3_lda=16.0_c5')   # n_classes = 5

#     exit(0)

