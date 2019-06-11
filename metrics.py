import torch
import torch.nn as nn


class LossUnsupervised(nn.Module):

    def __init__(self, num_clusters, feature_size=128):
        super(LossUnsupervised, self).__init__()

        self.m = nn.Parameter(torch.Tensor(num_clusters, feature_size))
        nn.init.xavier_uniform_(self.m)
    
        self.s1 = None; self.s2 = None
        
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
        total = intra / inter

        return total
