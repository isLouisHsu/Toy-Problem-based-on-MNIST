import math
import torch
import torch.nn as nn
from torch.nn import Parameter


class MarginProduct(nn.Module):
    """
    Notes:
        $$
        \text{softmax} = \frac{1}{N} \sum_i -\log \frac{e^{\tilde{y}_{y_i}}}{\sum_i e^{\tilde{y}_i}}
        $$

        $\text{where}$
        $$
        \tilde{y} = \begin{cases}
            s(\cos(m_1 \theta_{j, i} + m_2) + m_3) & j = y_i \\
            s(\cos(    \theta_{j, i}))             & j \neq y_i
        \end{cases}
        $$
    """

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35):

        super(MarginProduct, self).__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    def _acos(self, x):
        """
        Params:
            x: {tensor}
        Notes:
            防止梯度计算不稳定，arccos泰勒展开，近似计算
        """
        y = math.pi / 2 - x - x**3 / 6
        return y

    def _cosPhi(self, x):
        """
        Params:
            x: {tensor} [0, pi]
        Noets:
            由于$m_1*\theta + m_2 \in [m_2, m_1*pi + m_2]$在该区间内$cos(\theta)$不单调，故做相应处理
            $$
            \Phi(\theta) = \begin{cases}
                \cos (m_1*\theta + m_2) - m_3 & m_1*\theta + m_2 < \pi \\
                - 2 + \cos (m_1*\theta + m_2 - \pi) - m_3 & otherwise
            \end{cases}
            $$

            其函数图像可做出，代码如下
            ``` python
            import numpy as np
            import matplotlib.pyplot as plt

            def cosPhi(x, m1=1, m2=0.5, m3=0.35):

                phi = m1*x + m2
                cosPhi = np.cos(phi)
                y = np.where(phi < np.pi, cosPhi, - cosPhi - 2)
                y = y - m3

                return y


            if __name__ == '__main__':

                x = np.linspace(0, np.pi, 200)
                y = cosPhi(x, m1=1, m2=0.5, m3=0)

                plt.figure(0)
                plt.plot(x, y)
                plt.show()
            ```
        """
        phi = self.m1*x + self.m2
        cos_phi = torch.cos(phi)

        y = torch.where(phi > math.pi, - cos_phi - 2, cos_phi)
        y = y - self.m3

        return y

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
        theta  = self._acos(cosTheta)
        cosPhi = self._cosPhi(theta)
        output = torch.where(one_hot > 0, cosPhi, cosTheta)

        output = self.s * output
        
        return output


class MarginLoss(nn.Module):

    def __init__(self, s=32.0, m1=2.00, m2=0.50, m3=0.35):
        super(MarginLoss, self).__init__()

        self.margin = MarginProduct(s, m1, m2, m3)
        self.classifier = nn.CrossEntropyLoss()

    def forward(self, pred, gt):

        output = self.margin(pred, gt)
        loss   = self.classifier(output, gt)

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
