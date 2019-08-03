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


# 实验

## 设置lambda
``` python
for lamb in [5**i for i in range(6)]:    # 1, 5, 25, 125, 625, 3125
    main_unsupervised(3, 50, lamb=lamb, entropy_type='shannon', 
                        subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[lamb]{:4d}'.\
                                        format('shannon', 50, 3, lamb))
```

![1](/../images/1.jpg)

上图可见，调整`lamb`作用不大。

## 单位化x后，计算距离等

由于初始化各类中心为单位正交向量，故将特征先单位化
``` python
main_unsupervised(3, 50, entropy_type='shannon', 
                    subdir='unsupervised_{:s}_c{:3d}_f{:3d}_[normalized]'.\
                                    format('shannon', 50, 3))
```

# 修改为角度形式

## 类内熵分布：极小
$$
p^{(i)}_k = \frac{\exp( \frac{m_k^T x^{(i)}}{||m_k|| ||x^{(i)}||} )}{\sum_j \exp( \frac{m_j^T x^{(i)}}{||m_j|| ||x^{(i)}||} )}
$$

$$
L_{intra} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{C} p^{(i)}_k \log \frac{1}{p^{(i)}_k}
$$

其中$N$表示样本数，$C$表示指定的聚类数，$x^{(i)}$为第$i$个样本，$m_k$为第$k$类中心。

