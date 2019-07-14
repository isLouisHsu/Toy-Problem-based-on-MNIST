import math
import torch
import torch.nn as nn
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

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35, m4=0.50):

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
        cosPhi = self.m4 * (monocos(self.m1*theta + self.m2) - self.m3)
        
        output = torch.where(one_hot > 0, cosPhi, cosTheta)
        output = self.s * output
        
        return output


class MarginLoss(nn.Module):

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35, m4=0.50):
        super(MarginLoss, self).__init__()

        self.margin = MarginProduct(s, m1, m2, m3, m4)
        self.crossent = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.margin(pred, gt)
        loss   = self.crossent(output, gt)

        return loss


class MarginProductWithParameter(nn.Module):
    """
    Notes:
        based on ArcFace
    """

    def __init__(self, num_classes):

        super(MarginProductWithParameter, self).__init__()
        self.s  = Parameter(torch.ones(1)*8.0)
        self.m1 = Parameter(torch.ones(num_classes)*2.0)
        self.m2 = Parameter(torch.ones(num_classes)*0.5)
        self.m3 = Parameter(torch.ones(num_classes)*0.35)
        self.m4 = Parameter(torch.ones(num_classes)*0.5)

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

        m1, m2, m3, m4 = list(map(lambda x: x[label.long()].view(-1, 1), 
                                [self.m1, self.m2, self.m3, self.m4]))
        cosPhi = m4 * (monocos(m1 * theta + m2) - m3)
        
        output = torch.where(one_hot > 0, cosPhi, cosTheta)
        output = self.s * output
        
        return output


class MarginLossWithParameter(nn.Module):

    def __init__(self, num_classes):
        super(MarginLossWithParameter, self).__init__()

        self.margin = MarginProductWithParameter(num_classes)
        self.crossent = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.margin(pred, gt)
        loss   = self.crossent(output, gt)

        return loss


class LossUnsupervised(nn.Module):

    def __init__(self, num_clusters, feature_size=128):
        super(LossUnsupervised, self).__init__()

        self.m = nn.Parameter(torch.Tensor(num_clusters, feature_size))
        nn.init.xavier_uniform_(self.m)
    
        # self.s1 = None; self.s2 = None
        self.s1 = nn.Parameter(torch.ones(num_clusters))
        self.s2 = nn.Parameter(torch.ones(num_clusters))
        
    def _entropy(self, x):
        """
        Params:
            x: tensor{(n)}
        """
        x = torch.where(x<=0, 1e-16*torch.ones_like(x), x)
        x = torch.sum(- x * torch.log(x))
        return x

    def _softmax(self, x):
        """
        Params:
            x: {tensor(n)}
        """
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)

    def _f(self, x, m, s=None):
        """
        Params:
            x: {tensor(n_features(n))}
            m: {tensor(n_features(C, n))}
            s: {tensor(n_features(C))}
        Returns:
            y: {tensor(n_features(128))}
        Notes:
            p_{ik} = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
        """
        y = torch.norm(x - m, dim=1)
        if s is not None: y = y / s
            
        y = - y**2
        y = self._softmax(y)
        return y

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
        intra = torch.cat(list(map(lambda x: self._f(x, self.m, self.s1).unsqueeze(0), x)), dim=0)  # P_{N × n_classes} = [p_{ik}]
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), intra)), dim=0) # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                           # ent   = \frac{1}{N} \sum_i ent_i

        ## 类间
        inter = self._f(torch.mean(self.m, dim=0), self.m, self.s2)
        inter = self._entropy(inter)

        ## 优化目标，最小化
        # total = intra / inter
        # total = intra - inter
        total = intra + 1. / inter

        return total, intra, 1. / inter
