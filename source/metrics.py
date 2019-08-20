# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-07-11 11:15:04
@LastEditTime: 2019-08-20 08:54:19
@Update: 
'''
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils import softmax, norm

def arccos(x, n=5):
    """
    Params:
        x: {tensor}
        n: {int}
    """
    gt = lambda x, n: (math.factorial(2 * n - 1) /\
                        math.factorial(2 * n)) *\
                        (x**(2 * n + 1) / (2 * n + 1))
    y = math.pi / 2 - x
    for i in range(1, n):
        y -= gt(x, i)
    
    return y

def monocos(x):
    """
    Params:
        x: {tensor}
    Notes:
        y = \cos (x - n \pi) - 2n
        where n = \lfloor{} \frac{\theta}{\pi} \rfloor{}
    """
    n = x // math.pi
    y = torch.cos(x - n*math.pi) - 2*n

    return y

#############################################################################################
class MarginProduct(nn.Module):
    """
    Notes:
        $$
        \text{softmax} = \frac{1}{N} \sum_i -\log \frac{e^{\tilde{y}_{y_i}}}{\sum_i e^{\tilde{y}_i}}
        $$

        $\text{where}$
        $$
        \tilde{y} = \begin{cases}
            s(m4 \cos(m_1 \theta_{j, i} + m_2) + m_3) & j = y_i \\
            s(m4 \cos(    \theta_{j, i}))             & j \neq y_i
        \end{cases}
        $$
    """

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35, m4=2.00):

        super(MarginProduct, self).__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4

    def forward(self, cosTheta, label):
        """
        Params:
            cosTheta: {tensor(N, n_classes)} 每个样本(N)，到各类别(n_classes)矢量的余弦值
            label:  {tensor(N)}
        Returns:
            output: {tensor(N, n_classes)}
        """
        one_hot = torch.zeros(cosTheta.size(), device='cuda' if \
                        torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # theta  = torch.acos(cosTheta)
        theta  = arccos(cosTheta)
        cosPhi = self.m4 * (monocos(self.m1*theta + self.m2) - self.m3 - 1)
        
        output = torch.where(one_hot > 0, cosPhi, cosTheta - 1)
        output = self.s * output
        
        return output


class MarginLoss(nn.Module):

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35, m4=2.00):
        super(MarginLoss, self).__init__()

        self.margin = MarginProduct(s, m1, m2, m3, m4)
        self.crossent = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.margin(pred, gt)
        loss   = self.crossent(output, gt)

        return loss

# ===========================================================================================

class MarginProductWithParameter(nn.Module):
    """
    Notes:
        based on ArcFace
    """

    def __init__(self, num_classes, s=8.0):

        super(MarginProductWithParameter, self).__init__()

        self.s  = s
        
        self.m1 = Parameter(torch.ones(1)*1.00)
        self.m2 = Parameter(torch.ones(1)*0.00)
        self.m3 = Parameter(torch.ones(1)*0.00)
        self.m4 = Parameter(torch.ones(1)*1.00)

    def forward(self, cosTheta, label):
        """
        Params:
            cosTheta: {tensor(N, n_classes)} 每个样本(N)，到各类别(n_classes)矢量的余弦值
            label:  {tensor(N)}
        Returns:
            output: {tensor(N, n_classes)}
        """
        one_hot = torch.zeros(cosTheta.size(), device='cuda' if \
                        torch.cuda.is_available() else 'cpu')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # theta  = torch.acos(cosTheta)
        theta  = arccos(cosTheta)
        
        cosPhi = self.m4 * (monocos(self.m1 * theta + self.m2) - self.m3 - 1)
        
        output = torch.where(one_hot > 0, cosPhi, cosTheta - 1)
        output = self.s * output
        
        return output


class MarginLossWithParameter(nn.Module):

    def __init__(self, num_classes, s=8.0):
        super(MarginLossWithParameter, self).__init__()

        self.margin = MarginProductWithParameter(num_classes, s)
        self.crossent = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.margin(pred, gt)
        loss   = self.crossent(output, gt)

        return loss

#############################################################################################

class LossUnsupervisedEntropy(nn.Module):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):
        super(LossUnsupervisedEntropy, self).__init__()

        self.lamb = lamb
        self.entropy_type = entropy_type

        m = np.random.rand(num_clusters, feature_size)
        if num_clusters < feature_size:
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            m = vh[:]
        m /= np.linalg.norm(m, axis=1).reshape(-1, 1)
        self.m = nn.Parameter(torch.from_numpy(m).float())

        self.s1 = None; self.s2 = None
        # self.s1 = nn.Parameter(torch.ones(num_clusters)); self.s2 = nn.Parameter(torch.ones(num_clusters))

    def _p(self, x, m, s=None):
        """
        Params:
            x: {tensor(n_features(n_features))}
            m: {tensor(n_features(num_clusters, n_features))}
            s: {tensor(n_features(num_clusters))}
        Returns:
            y: {tensor(num_clusters}
        Notes:
            p^{(i)}_k = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
        """
        y = norm(x, m, s)
            
        y = - y**2
        y = softmax(y)
        return y
        
    def _entropy(self, p):
        """
        Params:
            p: tensor{(num_clusters)}
        """
        p = torch.where(p<=0, 1e-16*torch.ones_like(p), p)
        p = torch.where(p>=1, 1 - 1e-16*torch.ones_like(p), p)
        
        if self.entropy_type == 'shannon':
            p = - torch.sum(p * torch.log(p))
        elif self.entropy_type == 'kapur':
            p = - torch.sum(p * torch.log(p) + (1 - p) * torch.log(1 - p))

        return p

    def forward(self, x):
        """
        Params:
            x:    {tensor(N, n_features(128))}
        Returns:
            loss: {tensor(1)}
        Notes:
        -   p_{ik} = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
        -   entropy^{(i)}  = - \sum_k p_{ik} \log p_{ik}
        -   inter = \frac{1}{N} \sum_i entropy^{(i)}
        """
        ## 类内，属于各类别的概率的熵，求极小
        p = torch.cat(list(map(lambda x: self._p(x, self.m, self.s1).unsqueeze(0), x)), dim=0)      # P_{N × n_classes} = [p_{ik}]
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), p)), dim=0)             # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                                   # ent   = \frac{1}{N} \sum_i ent_i

        ## 类间
        p = self._p(torch.mean(self.m, dim=0), self.m, self.s2)
        inter = self._entropy(p)

        ## 优化目标，最小化
        # total = intra / inter
        total = intra - self.lamb * inter
        # total = intra + 1. / inter

        return total, intra, inter

# ===========================================================================================

class LossUnsupervisedWeightedSum(LossUnsupervisedEntropy):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):
        super(LossUnsupervisedWeightedSum, self).__init__(num_clusters, feature_size, lamb, entropy_type)

        self.s1 = None; self.s2 = None

    def forward(self, x):
        """
        Params:
            x:    {tensor(N, n_features(128))}
        Returns:
            loss: {tensor(1)}
        """
        ## 类内，属于各类别的概率的熵，求极小
        p = torch.cat(list(map(lambda x: self._p(x, self.m, self.s1).unsqueeze(0), x)), dim=0)  # P_{N × n_classes} = [p_{ik}]
        n = torch.cat(list(map(lambda x: norm(x, self.m).unsqueeze(0), x)), dim=0)              # N_{N × n_classes} = [n_{ik}]
        intra = torch.mean(torch.sum(p * n, dim=1))

        ## 类间
        m = torch.mean(self.m, dim=0)
        p = self._p(m, self.m, self.s2)
        n = norm(m, self.m)
        inter = torch.sum(p * n)

        ## 优化目标，最小化
        # total = intra / inter
        total = intra - self.lamb * inter
        # total = intra + 1. / inter

        return total, intra, inter

# ===========================================================================================

class LossReconstruct(nn.Module):

    def __init__(self):
        super(LossReconstruct, self).__init__()

    def forward(self, x, r):

        e = x - r
        e = e**2

        l = torch.sum(e.view(e.shape[0], -1), dim=1)
        l = torch.mean(l)

        return l


class LossUnsupervisedWithEncoderDecoder(nn.Module):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):
        super(LossUnsupervisedWithEncoderDecoder, self).__init__()
        
        self.reconstruct_loss = LossReconstruct()
        self.unsupervised_loss = LossUnsupervisedWeightedSum(num_clusters, feature_size, lamb, entropy_type)
    
    def forward(self, x, f, r):

        loss_r = self.reconstruct_loss(x, r)
        loss_u, intra, inter = self.unsupervised_loss(f)

        total = loss_r + loss_u

        return total, loss_r, intra, inter
        
# ===========================================================================================

class LossUnsupervisedSigmaI(LossUnsupervisedEntropy):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):

        super(LossUnsupervisedSigmaI, self).__init__(num_clusters, feature_size, lamb, entropy_type)
        
        self.s1 = None; self.s2 = None
    
    def forward(self, x):
        """
        Params:
            x:    {tensor(N, n_features(128))}
        Returns:
            loss: {tensor(1)}
        """
        N = x.shape[0]

        p = torch.cat(list(map(lambda x: self._p(x, self.m, self.s1).unsqueeze(0), x)), dim=0)      # P_{N × n_classes} = [p_{ik}]
        t = torch.sum(p, dim=0)

        p = p / N; t = t / N

        Lt = list(map(lambda x: self._entropy(x).unsqueeze(0), p))
        Lt = torch.sum(torch.cat(Lt, dim=0))

        inter = self._entropy(t)
        intra = Lt - inter

        total = intra - inter   # total = Lt - 2*inter

        return total, intra, inter
