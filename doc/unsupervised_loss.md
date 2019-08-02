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

# Kapur熵

$$ K(p) = - \sum_{k=1}^C p_k \log p_k + \frac{1}{\gamma} \sum_{k=1}^C (1 + \gamma p_k) \ln (1 + \gamma p_k) - \frac{1}{\gamma} (1 + \gamma) \ln (1 + \gamma) $$

其中 $-1 < \gamma < 1$
