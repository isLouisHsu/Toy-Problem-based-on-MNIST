import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, outp=128):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(   1,   32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(  32,   64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(  64,  128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 128,  256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d( 256, outp, 7),
            nn.Conv2d(outp, outp, 1),
        )

    def forward(self, x):

        x = self.layers(x)
        x = x.view(x.shape[0], -1)

        return  x

