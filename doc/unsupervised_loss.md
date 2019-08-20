# 2019.5
## 类内熵分布：极小
$$
p^{(i)}_k = \frac{\exp( - \frac{||x^{(i)} - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||x^{(i)} - m_j||^2}{s_j^2})} \tag{1.1}
$$

$$
L_{intra} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k} \tag{1.2}
$$

其中$N$表示样本数，$C$表示指定的聚类数，$x^{(i)}$为第$i$个样本，$m_k$为第$k$类中心。

## 类间熵分布：极大

各类聚类中心矢量的中心

$$ m = \frac{1}{C} \sum_{k=1}^C m_k \tag{2} $$

$$
p_k = \frac{\exp( - \frac{||m - m_k||^2}{s_k^2})}{\sum_j \exp( - \frac{||m - m_j||^2}{s_j^2})} \tag{2.1}
$$

$$
L_{inter} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p_k \log \frac{1}{p_k} \tag{2.2}
$$

## 总体损失

$$ L = \frac{L_{intra}}{L_{inter}} $$

$$ L = L_{intra} - L_{inter} $$

$$ L = L_{intra} + \frac{1}{L_{inter}} $$

# 2019.08.03

## Kapur熵

$$ K(p) = - \sum_{k=1}^C p_k \log p_k + \frac{1}{\gamma} \sum_{k=1}^C (1 + \gamma p_k) \ln (1 + \gamma p_k) - \frac{1}{\gamma} (1 + \gamma) \ln (1 + \gamma) $$

其中 $-1 < \gamma < 1$

## L
$$ L_{intra, weighted sum} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k ||x^{(i)} - m_k||^2 \tag{3.1} $$

$$ L_{inter, weighted sum} = \sum_{k=1}^{C} p_k ||m - m_k||^2 \tag{3.2} $$

$$ L = L_{intra} - L_{inter} \tag{3.3} $$

# 2019.08.18

## $p_{ij}$

由
$$ \min \left\{ \sum_i \sum_j p_{ij} \ln p_{ij} + \sum_i \sum_j \eta_j || x^{(i)} - m_j ||^2 p_{ij} \right\} $$

$$ \text{s.t.} \quad \sum_j p_{ij} = 1 $$

令
$$ \frac{\partial J(p_{ij})}{\partial p_{ij}} = \ln p_{ij} + 1 + \eta_j || x^{(i)} - m_j ||^2 = 0 $$

有
$$ p_{ij} = \exp (-1 - \eta_j || x^{(i)} - m_j ||^2) $$

$$ \Rightarrow p_{ij} = \frac{\exp (- \eta_j || x^{(i)} - m_j ||^2)}{\sum_k \exp - \eta_k || x^{(i)} - m_k ||^2}$$

## $L$
<!-- $$ L_{intra} = - \frac{1}{N} \sum_j \sum_i \frac{p_{ij}}{\sum_i p_{ij}} \ln \frac{p_{ij}}{\sum_i p_{ij}} $$ -->

定义
$$ L_{t} = - \sum_i \sum_j \frac{p_{ij}}{\sum_i \sum_j p_{ij}} \ln \frac{p_{ij}}{\sum_i \sum_j p_{ij}} $$

其中$\sum_j p_{ij} = 1$，故$\sum_i \sum_j p_{ij} = N$

$$ L_{t} = - \sum_i \sum_j \frac{p_{ij}}{N} \ln \frac{p_{ij}}{N} $$

令$T_j = \sum_i p_{ij}$，有$\sum_j T_j = N$

$$ L_{t} = - \sum_i \sum_j \frac{p_{ij}}{N} \frac{T_j}{T_j} \ln \frac{p_{ij}}{N} \frac{T_j}{T_j} $$

$$ = - \sum_i \sum_j \frac{p_{ij}}{T_j} \frac{T_j}{N} (\ln \frac{p_{ij}}{T_j} + \ln \frac{T_j}{N}) $$

$$ = \underbrace{- \sum_j \frac{T_j}{N} \sum_i \frac{p_{ij}}{T_j} \ln \frac{p_{ij}}{T_j}}_{L_{intra}} \underbrace{- \sum_j \frac{T_j}{N} \ln \frac{T_j}{N}}_{L_{inter}} $$

> $$ L_{intra} = - \sum_i \sum_j \frac{p_{ij}}{N} \ln \frac{p_{ij}}{N} + \sum_j \frac{T_j}{N} \ln \frac{T_j}{N} $$

记
$$ L = L_{intra} - L_{inter} = - \sum_i \sum_j \frac{p_{ij}}{N} \ln \frac{p_{ij}}{N} + 2 \sum_j \frac{T_j}{N} \ln \frac{T_j}{N} $$