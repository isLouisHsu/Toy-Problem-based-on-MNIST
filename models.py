import torch
import torch.nn as nn


class NetSupervised(nn.Module):

    def __init__(self, num_classes):
        super(NetSupervised, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(   1,   32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(  32,   64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(  64,  128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 128,  256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 256,  256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d( 256,  256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 256, num_classes, 7),
            nn.Conv2d(num_classes, num_classes, 1),
        )

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.shape[0], -1)

        return  x


class NetUnsupervised(nn.Module):

    def __init__(self, num_clusters, feature_size=128):
        super(NetUnsupervised, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(   1,   32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(  32,   64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(  64,  128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 128,  256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 256,  256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d( 256,  256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 256, feature_size, 7),
            nn.Conv2d(feature_size, feature_size, 1),
        )

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.shape[0], -1)

        return  x
