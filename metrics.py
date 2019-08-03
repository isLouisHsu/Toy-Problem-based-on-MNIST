import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

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

class LossUnsupervised(nn.Module):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):
        super(LossUnsupervised, self).__init__()

        self.lamb = lamb
        self.entropy_type = entropy_type

        m = np.random.rand(num_clusters, feature_size)
        if num_clusters > feature_size: m = m.T
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        m = vh[: num_clusters]
        if num_clusters > feature_size: m = m.T
        self.m = nn.Parameter(torch.from_numpy(m).float())

        # self.s1 = None; self.s2 = None
        self.s1 = nn.Parameter(torch.ones(num_clusters))
        self.s2 = nn.Parameter(torch.ones(num_clusters))

    def _softmax(self, x):
        """
        Params:
            x: {tensor(n_samples)}
        Returns:
            x: {tensor(n_samples)}
        Notes:
            - x_i := \frac{e^{x_i}} {\sum_j e^{x_j}}
        """
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)

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
        y = torch.norm(x - m, dim=1)
        if s is not None: y = y / s
            
        y = - y**2
        y = self._softmax(y)
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
        ## TODO: x单位化 or batchnorm
        x = F.normalize(x)

        ## 类内，属于各类别的概率的熵，求极小
        intra = torch.cat(list(map(lambda x: self._p(x, self.m, self.s1).unsqueeze(0), x)), dim=0)  # P_{N × n_classes} = [p_{ik}]
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), intra)), dim=0)         # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                                   # ent   = \frac{1}{N} \sum_i ent_i

        ## 类间
        inter = self._p(torch.mean(self.m, dim=0), self.m, self.s2)
        inter = self._entropy(inter)

        ## 优化目标，最小化
        # total = intra / inter
        total = intra - self.lamb * inter
        # total = intra + 1. / inter

        return total, intra, inter


class LossUnsupervisedAngle(nn.Module):

    def __init__(self, num_clusters, feature_size, lamb=1.0, entropy_type='shannon'):
        super(LossUnsupervisedAngle, self).__init__()

        self.lamb = lamb
        self.entropy_type = entropy_type

        m = np.random.rand(num_clusters, feature_size)
        if num_clusters > feature_size: m = m.T
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        m = vh[: num_clusters]
        if num_clusters > feature_size: m = m.T
        self.m = nn.Parameter(torch.from_numpy(m).float())

    def _softmax(self, x):
        """
        Params:
            x: {tensor(n_samples)}
        Returns:
            x: {tensor(n_samples)}
        """
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)

    def _p(self, x, m):
        """
        Params:
            x: {tensor(n_features(n_samples, n_features))}
            m: {tensor(n_features(num_clusters, n_features))}
        Returns:
            y: {tensor(n_samples, num_clusters}
        """
        y = F.linear(F.normalize(x), F.normalize(self.m))
        y = torch.cat(list(map(lambda x: self._softmax(x).unsqueeze(0), y)), dim=0)
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
        """

        ## 类内，属于各类别的概率的熵，求极小
        intra = self._p(x, self.m)
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), intra)), dim=0)         # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                                   # ent   = \frac{1}{N} \sum_i ent_i
        
        inter = 0
        total = intra

        return total, intra, inter