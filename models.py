import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Network(nn.Module):

    def __init__(self, num_classes, feature_size):
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

    def __init__(self, num_classes, feature_size):
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

    def get_feature(self, x):
        
        x = self.pre_layers(x)
        x = x.view(x.shape[0], -1)

        return x

    def forward(self, x):

        x = self.get_feature(x)
        x = self.cosine_layer(x)

        return  x


if __name__ == "__main__":
    net = NetworkMargin(10, 2)
    X = torch.rand(32, 1, 28, 28)
    net(X)