import os
import time
import torch
import numpy as np

getTime     = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def accuracy(y_pred_prob, y_true):
    """
    Params:
        y_pred_prob:{tensor(N, n_classes) or tensor(N, C, n_classes)}
        y_true:     {tensor(N)}
    Returns:
        acc:        {tensor(1)}
    """
    y_pred = torch.argmax(y_pred_prob, 1)
    acc = torch.mean((y_pred==y_true).float())
    return acc

def softmax(x):
    """
    Params:
        x: {tensor(n_classes)}
    Returns:
        x: {tensor(n_classes)}
    Notes:
        - x_i := \frac{e^{x_i}} {\sum_j e^{x_j}}
    """
    x = torch.exp(x - torch.max(x))
    return x / torch.sum(x)

def norm(x1, x2, s=None):
    y = torch.norm(x1 - x2, dim=1)
    if s is not None: y = y / s
    return y