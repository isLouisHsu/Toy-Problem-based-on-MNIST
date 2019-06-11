import torch
import torch.nn as nn


class NetSupervised(nn.Module):

    def __init__(self, num_classes):
        super(NetSupervised, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(   1,  32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(  32,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d( 64,  128, 7),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.shape[0], -1)

        return  x


class NetUnsupervised(nn.Module):

    def __init__(self, feature_size=128):
        super(NetUnsupervised, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(   1,  32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(  32,  64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d( 64,  128, 7),
            nn.Conv2d(128, feature_size, 1),
        )

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.shape[0], -1)

        return  x

if __name__ == "__main__":
    net = NetSupervised(10)
    X = torch.rand(32, 1, 28, 28)
    net(X)