# 类内熵分布：极小
$$
p^{(i)}_k = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
$$

$$
L_{intra} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k}
$$

其中$N$表示样本数，$C$表示指定的聚类数，$x^{(i)}$为第$i$个样本，$m_k$为第$k$类中心。

# 类间熵分布：极大

各类聚类中心矢量的中心

$$ m = \frac{1}{C} \sum_{k=1}^C m_k $$

$$
p^{(i)}_k = \frac{\exp( - \frac{||m - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||m - m_j||^2}{s_j^2})}
$$

$$
L_{inter} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k}
$$

# 总体损失

尝试了三种形式

$$ L = \frac{L_{intra}}{L_{inter}} $$

$$ L = L_{intra} - L_{inter} $$

$$ L = L_{intra} + \frac{1}{L_{inter}} $$

``` python
class LossUnsupervised(nn.Module):
    """
    Attributes:
        -   m:  {tensor(num_clusters, feature_size)} 每个类别的中心矢量矩阵
        -   s1: {tensor(num_clusters)}
        -   s2: {tensor(num_clusters)}
    Notes:
        -   
    """
    def __init__(self, num_clusters, feature_size=128):
        super(LossUnsupervised, self).__init__()

        self.m = nn.Parameter(torch.Tensor(num_clusters, feature_size))
        nn.init.xavier_uniform_(self.m)
    
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
            x: {tensor(n_features(n_samples))}
            m: {tensor(n_features(num_clusters, n_samples))}
            s: {tensor(n_features(num_clusters))}
        Returns:
            y: {tensor(n_features(128))}
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
        Notes(markdown):
            # 类内熵分布：极小
            $$
            p^{(i)}_k = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})}
            $$

            $$
            L_{intra} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k}
            $$

            其中$N$表示样本数，$C$表示指定的聚类数，$x^{(i)}$为第$i$个样本，$m_k$为第$k$类中心。

            # 类间熵分布：极大

            各类聚类中心矢量的中心

            $$ m = \frac{1}{C} \sum_{k=1}^C m_k $$

            $$
            p^{(i)}_k = \frac{\exp( - \frac{||m - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||m - m_j||^2}{s_j^2})}
            $$

            $$
            L_{inter} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k}
            $$

            # 总体损失

            尝试了三种形式

            $$ L = \frac{L_{intra}}{L_{inter}} $$

            $$ L = L_{intra} - L_{inter} $$

            $$ L = L_{intra} + \frac{1}{L_{inter}} $$
        """
        p = torch.where(p<=0, 1e-16*torch.ones_like(p), p)
        p = torch.sum(- p * torch.log(p))
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
        intra = torch.cat(list(map(lambda x: self._p(x, self.m, self.s1).unsqueeze(0), x)), dim=0)  # P_{N × n_classes} = [p_{ik}]
        intra = torch.cat(list(map(lambda x: self._entropy(x).unsqueeze(0), intra)), dim=0)         # ent_i = \sum_k p_{ik} \log p_{ik}
        intra = torch.mean(intra)                                                                   # ent   = \frac{1}{N} \sum_i ent_i

        ## 类间
        inter = self._p(torch.mean(self.m, dim=0), self.m, self.s2)
        inter = self._entropy(inter)

        ## 优化目标，最小化
        # total = intra / inter
        # total = intra - inter
        total = intra + 1. / inter

        return total, intra, 1. / inter
```
