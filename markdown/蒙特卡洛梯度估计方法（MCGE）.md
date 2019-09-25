# 蒙特卡洛梯度估计方法（MCGE）

## 基本思想

机器学习中最常见的优化算法是基于梯度的优化方法，当目标函数是一个类似如下结构的随机函数$F(\theta)$ 时：
$$
F(\theta)=\int p(x ; \theta) f(x ; \phi)=E_{p(x ; \theta)}[f(x ; \phi)]
$$
优化该类目标函数，最核心的计算问题是对随机函数$$F(θ)$$ 的梯度进行估计，即：
$$
\eta=\nabla_{\theta} F(\theta)=\nabla_{\theta} E_{p(x ; \theta)}[f(x ; \phi)]
$$


公式（1）中的积分内是一个分布和代价函数的乘积，在对其梯度进行近似估计时，可以从两个方面进行求导。由此，**可以将梯度估计方法大致分为两类：**

- **求解分布测度的导数**，包括本文介绍的 score function gradient estimator
- **求解代价函数的导数**，包括本文介绍的 pathwise gradient estimator



使用蒙特卡洛估计的时候**往往关注其以下四个性质：**

- 一致性，根据大数定理，当所采样的样本数量非常多时，MCE 的估计值将会收敛到积分的真值处。

- 无偏性，MCE 是对所求积分的一个无偏估计，简单推导如下：
  $$
  \mathbb{E}_{p(\mathbf{x} ; \boldsymbol{\theta})}\left[\overline{\mathcal{F}}_{N}\right]=\mathbb{E}_{p(\mathbf{x} ; \boldsymbol{\theta})}\left[\frac{1}{N} \sum_{n=1}^{N} f\left(\mathbf{x}^{(n)}\right)\right]=\frac{1}{N} \sum_{n=1}^{N} \mathbb{E}_{p(\mathbf{x} ; \boldsymbol{\theta})}\left[f\left(\mathbf{x}^{(n)}\right)\right]=\mathbb{E}_{p(\mathbf{x} ; \boldsymbol{\theta})}[f(\mathbf{x})]
  $$
  

​					MCE 的无偏性是随机优化算法收敛的重要保证。

- 小方差，当几个估计方法都是无偏估计时，我们通常会选择方差较小的 MCE，因为更小方差的 MCE 会估计地更准，从而使得优化地效率更高、准确性更好。

- 可计算性，很多机器学习问题都是高维问题，如何提高 MCE 的可计算性，比如：减少采样、提高并行能力等变得十分重要。

  

## Score Function Gradient Estimator

有些时候，我们的目标函数不可导，在这种情况下，我们就没有办法求解$\eta$ 的值。

因此我们用SFGE的方法来将期望的梯度变换为梯度的期望（这样就不用对$f(x)$ 求梯度了），从而实现梯度的近似估计。
$$
\begin{array}{l}{\nabla_{\theta} E_{p(x ; \theta)}[f(x)]} \\ {=\nabla_{\theta} \int p(x ; \theta) f(x) d x} \\ {=\int f(x) \nabla_{\theta} p(x ; \theta) d x} \\ {=\int p(x ; \theta) f(x) \nabla_{\theta} \log p(x ; \theta) d x} \\ {=E_{p(x ; \theta)}\left[f(x) \nabla_{\theta} \log p(x ; \theta)\right]}\end{array}
$$
这中间有一个过程是将积分和微分操作的位置进行了对换，此操作并非可以随意进行，需要满足一定的条件，但一般的机器学习问题都会满足。

#### SFGE的性质

- 代价函数$f(x)$ 可以是任意函数。比如可微的，不可微的；离散的，连续的；白箱的，黑箱的等。这个性质是其最大的优点，使得很多不可微的甚至没有具体函数的黑箱优化问题都可以利用梯度优化求解。
- 分布函数$$p(x;\theta)$$ 必须对 *θ* 是可微的。
- 分布函数必须是便于采样的，因为梯度估计都是基于 MC 的，所以希望分布函数便于采样。
- SFGE 的方差受很多因素影响，包括输入的维度和代价函数。

#### SFGE的典型应用

SFGE 由于其对代价函数没有限制，具有非常广阔的应用场景，以下是几个非常热门的应用：

- 策略梯度优化算法 REINFORCE 及其变种

- 基于 GAN 的自然语言生成

- 基于自动微分的黑盒变分推断

  

## Pathwise Gradient Estimator 

不同于 SFGE 对代价函数没有任何约束，PGE 要求代价函数可微。

核心是做分布变换（即所谓的 reparameterization ，重参数化），计算原来分布下的期望梯度时，由于变换后的分布不包含求导参数，可将求导和积分操作进行对换，从而基于 MC 对梯度进行估计。
$$
\nabla_{\theta} \mathbb{E}_{p(z ; \theta)}[f(z)]=\mathbb{E}_{p(\epsilon)}\left[\nabla_{\theta} f(g(\epsilon, \theta))\right]
$$
由于，$p(z)=\left|\frac{d \epsilon}{d z}\right| p(\epsilon) \Longrightarrow|p(z) d z|=|p(\epsilon) d \epsilon|$ 

因此我们可以用$p(\epsilon)$ 来替代$p(z)$，将原式重写可得
$$
\begin{aligned} & \nabla_{\theta} \mathbb{E}_{p(z, \theta)}[f(z)]=\nabla_{\theta} \int p(z ; \theta) f(z) d z \\=& \nabla_{\theta} \int p(\epsilon) f(z) d \epsilon=\nabla_{\theta} \int p(\epsilon) f(g(\epsilon, \theta)) d \epsilon \\=& \nabla_{\theta} \mathbb{E}_{p(\epsilon)}[f(g(\epsilon, \theta))]=\mathbb{E}_{p(\varepsilon)}\left[\nabla_{\theta} f(g(\epsilon, \theta))\right] \end{aligned}
$$
就可以利用蒙特卡洛估计的方法求得
$$
\mathbb{E}_{p(\epsilon)}\left[\nabla_{\theta} f(g(\epsilon, \theta))\right]=\frac{1}{S} \sum_{s=1}^{S} \nabla_{\theta} f\left(g\left(\epsilon^{(s)}, \theta\right)\right), \quad \epsilon^{(s)} \sim p(\epsilon)
$$
从推导中可以看出，分布中的参数被 push 到了代价函数中，从而可以将求导和积分操作进行对换。

分布变换是统计学中一个基本的操作，在计算机中实际产生各种常见分布的随机数时，都是基于均匀分布的变换来完成的。有一些常见的分布变换可参见下表：

<img src="D:/学习/github/RL_note/picture/Fisher/1.png" style="zoom:95%"/>

