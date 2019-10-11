# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-11 11:15:04
@LastEditTime: 2019-10-11 11:36:01
@Update: 
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Network(nn.Module):

    def __init__(self, num_classes, feature_size, init_once=True):
        super(Network, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(   1,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(  64,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d( 64,  feature_size, 7),
            nn.Conv2d(feature_size, num_classes, 1),
        )

        if init_once:
            pkldir = '../initial'
            if not os.path.exists(pkldir): os.mkdir(pkldir)
            pklpath = '%s/%s_class%d_dim%d.pkl' % (pkldir, self._get_name(), num_classes, feature_size)
            if os.path.exists(pklpath):
                state = torch.load(pklpath)
                self.load_state_dict(state)
            else:
                state = self.state_dict()
                torch.save(state, pklpath)

    def get_feature(self, x):
        
        x = self.pre_layers(x)

        return x

    def forward(self, x):

        x = self.pre_layers(x)
        x = x.view(x.shape[0], -1)
        
        return  x


class CosineLayer(nn.Module):
    """
    Attributes:
        weight: {Parameter(num_classes, feature_size)}
    """
    def __init__(self, num_classes, feature_size):
        super(CosineLayer, self).__init__()
        
        self.weights = Parameter(torch.Tensor(num_classes, feature_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        """
        Params:
            x: {tensor(N, feature_size)}
        Notes:
            \cos \theta^{(i)}_j = \frac{W_j^T f^{(i)}}{||W_j|| ||f^{(i)}||}
        """
        x = F.linear(F.normalize(x), F.normalize(self.weights))
        return x

class NetworkMargin(nn.Module):

    def __init__(self, num_classes, feature_size, init_once=True):
        super(NetworkMargin, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv2d(   1,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(  64,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d( 64,  feature_size, 7),
        )

        self.cosine_layer = CosineLayer(num_classes, feature_size)

        if init_once:
            pkldir = '../initial'
            if not os.path.exists(pkldir): os.mkdir(pkldir)
            pklpath = '%s/%s_class%d_dim%d.pkl' % (pkldir, self._get_name(), num_classes, feature_size)
            if os.path.exists(pklpath):
                state = torch.load(pklpath)
                self.load_state_dict(state)
            else:
                state = self.state_dict()
                torch.save(state, pklpath)

    def get_feature(self, x):
        
        x = self.pre_layers(x)
        x = x.view(x.shape[0], -1)

        return x

    def forward(self, x):

        x = self.get_feature(x)
        x = self.cosine_layer(x)

        return  x


class NetworkUnsupervised(nn.Module):

    def __init__(self, feature_size, init_once=True, batchnorm=False):
        super(NetworkUnsupervised, self).__init__()

        if batchnorm:
            self.pre_layers = nn.Sequential(
                nn.Conv2d(   1,  64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(  64,  64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d( 64,  feature_size, 7),
                nn.BatchNorm2d(feature_size),
            )
        else:
            self.pre_layers = nn.Sequential(
                nn.Conv2d(   1,  64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(  64,  64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d( 64,  feature_size, 7),
            )

        if init_once:
            pkldir = '../initial'
            if not os.path.exists(pkldir): os.mkdir(pkldir)
            pklpath = '%s/%s_dim%d%s.pkl' % (pkldir, 
                    self._get_name(), feature_size, '_bn' if batchnorm else '')
            if os.path.exists(pklpath):
                state = torch.load(pklpath)
                self.load_state_dict(state)
            else:
                state = self.state_dict()
                torch.save(state, pklpath)

    def get_feature(self, x):
        
        x = self.pre_layers(x)
        x = x.view(x.shape[0], -1)

        return x

    def forward(self, x):

        x = self.get_feature(x)
        
        return  x


class NetworkUnsupervisedWithEncoderDecoder(nn.Module):

    def __init__(self, feature_size, init_once=True):
        super(NetworkUnsupervisedWithEncoderDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(   1,  64, 3, 1, 1),          # 28 x 28 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     # 14 x 14 x 64

            nn.Conv2d(  64,  64, 3, 1, 1),          # 14 x 14 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     #  7 x  7 x 64

            nn.Conv2d( 64,  feature_size, 3, 1, 1), #  7 x  7 x feature_size

            nn.AdaptiveAvgPool2d((1, 1)),           #  1 x  1 x feature_size
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=7),#  7 x  7 x feature_size

            nn.Conv2d(feature_size, 64, 3, 1, 1),   #  7 x  7 x 64
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),# 14 x 14 x 64

            nn.Conv2d(64, 64, 3, 1, 1),             # 14 x 14 x 64
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),# 28 x 28 x 64
            
            nn.Conv2d(64,  1, 3, 1, 1),             # 28 x 28 x 1
        )

        if init_once:
            pkldir = '../initial'
            if not os.path.exists(pkldir): os.mkdir(pkldir)
            pklpath = '%s/%s_dim%d.pkl' % (pkldir, self._get_name(), feature_size)
            if os.path.exists(pklpath):
                state = torch.load(pklpath)
                self.load_state_dict(state)
            else:
                state = self.state_dict()
                torch.save(state, pklpath)
    
    def get_feature(self, x):

        x = self.encoder(x)
        x = x.view(x.shape[0], -1)

        return x

    def get_reconstruct(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def forward(self, x):

        f = self.get_feature(x)
        r = self.get_reconstruct(x)

        return f, r


class GeneratorNet(nn.Module):

    def __init__(self, feature_size):
        super(GeneratorNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=7),    #  7 x  7 x feature_size

            nn.Conv2d(feature_size, 64, 3, 1, 1),       #  7 x  7 x 64
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),    # 14 x 14 x 64

            nn.Conv2d(64, 64, 3, 1, 1),                 # 14 x 14 x 64
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),    # 28 x 28 x 64
            
            nn.Conv2d(64,  1, 3, 1, 1),                 # 28 x 28 x 1
        )

    def forward(self, x):
        """ 
        Params:
            x: {tensor(N, D)}
        Returns:
            x: {tensor(N, 1, 28, 28)}
        """
        x = x.unsqueeze(-1).unsqueeze(-1)               # N x D x  1 x  1
        x = self.layers(x)                              # N x 1 x 28 x 28

        return x

class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(   1,  64, 3, 1, 1),          # 28 x 28 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     # 14 x 14 x 64

            nn.Conv2d(  64,  64, 3, 1, 1),          # 14 x 14 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                     #  7 x  7 x 64

            nn.Conv2d( 64,  128, 3, 1, 1),          #  7 x  7 x 128

            nn.AdaptiveAvgPool2d((1, 1)),           #  1 x  1 x 128
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """ 
        Params:
            x: {tensor(N, 1, 28, 28)}
        Returns:
            x: {tensor(N)}
        """
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = x.view(x.shape[0])

        return x


if __name__ == "__main__":
    net = GeneratorNet(32)
    net = DiscriminatorNet()
    X = torch.rand(128, 1, 28, 28)
    y = net(X)
    pass
