import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from config import configer
from datasets import MNIST
from metrics import LossUnsupervised, MarginLoss, MarginLossWithParameter
from models import Network, NetworkMargin
from trainer import SupervisedTrainer, UnsupervisedTrainer, MarginTrainer, MarginTrainerWithParameter, MarginTrainerWithVectorLoss

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

def main_adaptivemargin(num_classes=10, feature_size=2, s=8.0, each_class=False, subdir=None):
    net = NetworkMargin(num_classes=num_classes, feature_size=feature_size)
    criterion = MarginLossWithParameter(num_classes, s, each_class)

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

def main_margin_with_opposite_loss(num_classes=10, feature_size=2, s=8.0, m1=2.00, m2=0.5, m3=0.35, m4=0.5, lda=0.2, subdir=None):
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

    trainer = MarginTrainerWithVectorLoss(configer, net, params, trainset, validset, criterion, lda,
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

# ==============================================================================================================================
# if __name__ == "__main__":

#     # cosmulface
#     main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=2.00, subdir='cosmulface_dim2_m4=2.00')
#     main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=3.00, subdir='cosmulface_dim2_m4=3.00')
#     main_margin(num_classes=10, feature_size=2, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=4.00, subdir='cosmulface_dim2_m4=4.00')
#     # adaptiveface
#     main_adaptivemargin(num_classes=10, feature_size=2, s=8.0, each_class=False, subdir='adaptiveface_dim2_F')
#     main_adaptivemargin(num_classes=10, feature_size=2, s=8.0, each_class=True,  subdir='adaptiveface_dim2_T')

#     # cosmulface
#     main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=2.00, subdir='cosmulface_dim3_m4=2.00')
#     main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=3.00, subdir='cosmulface_dim3_m4=3.00')
#     main_margin(num_classes=10, feature_size=3, s=8.0, m1=1.00, m2=0.0, m3=0.00, m4=4.00, subdir='cosmulface_dim3_m4=4.00')
#     # adaptiveface
#     main_adaptivemargin(num_classes=10, feature_size=3, s=8.0, each_class=False, subdir='adaptiveface_dim3_F')
#     main_adaptivemargin(num_classes=10, feature_size=3, s=8.0, each_class=True,  subdir='adaptiveface_dim3_T')

#     exit(0)

# ==============================================================================================================================
if __name__ == "__main__":

    # arcface
    main_margin_with_opposite_loss(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.2, subdir='arcface_dim2_m2=0.5_lda=0.2')
    main_margin_with_opposite_loss(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.4, subdir='arcface_dim2_m2=0.5_lda=0.4')
    main_margin_with_opposite_loss(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.6, subdir='arcface_dim2_m2=0.5_lda=0.6')
    main_margin_with_opposite_loss(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.8, subdir='arcface_dim2_m2=0.5_lda=0.8')

    main_margin_with_opposite_loss(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.2, subdir='arcface_dim3_m2=0.5_lda=0.2')
    main_margin_with_opposite_loss(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.4, subdir='arcface_dim3_m2=0.5_lda=0.4')
    main_margin_with_opposite_loss(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.6, subdir='arcface_dim3_m2=0.5_lda=0.6')
    main_margin_with_opposite_loss(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, lda=0.8, subdir='arcface_dim3_m2=0.5_lda=0.8')

    exit(0)

## ==============================================================================================================================
if __name__ == "__main__":

    # modified
    main_margin(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_dim2')
    # sphereface
    main_margin(num_classes=10, feature_size=2, s= 8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_dim2_m1=2.00')
    # arcface
    main_margin(num_classes=10, feature_size=2, s=16.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=16')
    main_margin(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=8')
    main_margin(num_classes=10, feature_size=2, s= 4.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=4')
    main_margin(num_classes=10, feature_size=2, s= 1.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim2_m2=0.5_s=1')
    # cosface
    main_margin(num_classes=10, feature_size=2, s= 8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_dim2_m3=0.35')

    # modified
    main_margin(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.0, m3=0.00, m4=1.0, subdir='modified_dim3')
    # sphereface
    main_margin(num_classes=10, feature_size=3, s= 8.0, m1=2.00, m2=0.0, m3=0.00, m4=1.0, subdir='sphereface_dim3_m1=2.00')
    # arcface
    main_margin(num_classes=10, feature_size=3, s=16.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=16')
    main_margin(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=8')
    main_margin(num_classes=10, feature_size=3, s= 4.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=4')
    main_margin(num_classes=10, feature_size=3, s= 1.0, m1=1.00, m2=0.5, m3=0.00, m4=1.0, subdir='arcface_dim3_m2=0.5_s=1')
    # cosface
    main_margin(num_classes=10, feature_size=3, s= 8.0, m1=1.00, m2=0.0, m3=0.35, m4=1.0, subdir='cosface_dim3_m3=0.35')

    exit(0)
