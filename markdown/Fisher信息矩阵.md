# Fisher信息矩阵

## Score Function

一般来说我们会根据MLE（极大似然估计）来估计参数，为了方便计算一般都会采用对数似然（loglikelihood）当目标函数：
$$
max_\theta \text{log}p(x;\theta)
$$
记$L(\theta)=\text{log}p(x;\theta)$ ，求最优值最直接的方法就是令它的导数为0。即：
$$
\frac{dL(\theta)}{d\theta}=0
$$
这里$L(\theta)$ 的一阶导数即为Score Function，记为：
$$
S(\theta)=\frac{dL(\theta)}{d\theta}=\nabla_{\theta} \log p(x | \theta)
$$

因此求 MLE 相当于求解 Score 方程。

Score Function具有一些比较好的性质：

- 与MLE相关

- 其期望等于0，推导为：
  $$
  \begin{array}{l}{E_{p(x ; \theta)}\left[\nabla_{\theta} \log p(x ; \theta)\right]} \\ {=\int p(x ; \theta) \nabla_{\theta} \log p(x ; \theta)} \\ {=\int p(x ; \theta) \frac{\nabla_{\theta} p(x ; \theta)}{p(x ; \theta)}} \\ {=\int \nabla_{\theta} p(x ; \theta)} \\ {=\nabla_{\theta} \int p(x ; \theta)} \\ {=\nabla_{\theta} 1} \\ {=0}\end{array}
  $$
  这样会带来非常多的便利，比如：一种降低估计方差的思路，将代价函数$f(x)$ 变为$f(x)-b$ ，其中$b$ 就是baseline。这样可以令原来的公式在均值不改变的情况下方差降低。
  

$$
\begin{aligned} E_{p(x ; \theta)} \nabla_{\theta} \log p(x ; \theta)[f(x)-b] &=E_{p(x ; \theta)} \nabla_{\theta} \log p(x ; \theta) f(x)-b E_{p(x ; \theta)} \nabla_{\theta} \log p(x ; \theta) \\=& E_{p(x ; \theta)} \nabla_{\theta} \log p(x ; \theta) f(x) \end{aligned}
$$

### Score Function Gradient Estimator

机器学习中最常见的优化算法是基于梯度的优化方法，当目标函数是一个类似如下结构的随机函数$F(\theta)$ 时：
$$
F(\theta)=\int p(x ; \theta) f(x ; \phi)=E_{p(x ; \theta)}[f(x ; \phi)]
$$
优化该类目标函数，最核心的计算问题是对随机函数$$F(θ)$$ 的梯度进行估计，即：
$$
\eta=\nabla_{\theta} F(\theta)=\nabla_{\theta} E_{p(x ; \theta)}[f(x ; \phi)]
$$
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







我们对这个结果有多确信呢？这个时候就应该计算我们模型score的方差。
$$
\underset{p(x | \theta)}{\mathbb{E}}\left[(s(\theta)-0)(s(\theta)-0)^{\mathrm{T}}\right]
$$
我们可以把它当成是一个信息，这就是Fisher信息。假设$\theta$ 是一个vector，那么Fisher information就是矩阵形式，即Fisher information matrix。
$$
\mathrm{F}=\underset{p(x | \theta)}{\mathbb{E}}\left[\nabla \log p(x | \theta) \nabla \log p(x | \theta)^{\mathrm{T}}\right]
$$
然而实际上，likelihood function经常特别复杂而且难以计算期望。这个时候我们可以用蒙特卡洛的方法用经验分布$\hat{q}(x)$ 去近似$F$ ，这时候就有了训练集$X=\{x_1,x_2,...,x_N\}$ 。这时候，$F$ 被称为Empirical Fisher：
$$
\mathrm{F}=\frac{1}{N} \sum_{i=1}^{N} \nabla \log p\left(x_{i} | \theta\right) \nabla \log p\left(x_{i} | \theta\right)^{\mathrm{T}}
$$

Fisher information实际上也是Score function的负导数，即：
$$
I(\theta)=-\frac{d^{2} L(\theta)}{d \theta^{2}}=-\frac{d S(\theta)}{d \theta}
$$
Fisher information matrix还是log likelihood的Hessian矩阵的负期望。log likelihood的Hessian矩阵是根据其梯度的Jacobian矩阵得来的：
$$
\begin{aligned} \mathrm{H}_{\mathrm{log} p(x | \theta)} &=\mathrm{J}\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right) \\ &=\frac{\mathrm{H}_{p(x | \theta)} p(x | \theta)-\nabla p(x | \theta) \nabla p(x | \theta)^{\mathrm{T}}}{p(x | \theta) p(x | \theta)} \\ &=\frac{\mathrm{H}_{p(x | \theta)} p(x | \theta)}{p(x | \theta) p(x | \theta)}-\frac{\nabla p(x | \theta) \nabla p(x | \theta)^{\mathrm{T}}}{p(x | \theta) p(x | \theta)} \\ &=\frac{\mathrm{H}_{p(x | \theta)}}{p(x | \theta)}-\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)^{\mathrm{T}} \end{aligned}
$$
where the second line is a result of applying quotient rule of derivative.

对Hessian矩阵求期望：
$$
\begin{aligned} \mathbb{E}_{P(x | \theta)}\left[\mathrm{H}_{\log p(x | \theta)}\right] &=\underset{p(x | \theta)}{\mathbb{E}}\left[\frac{\mathrm{H}_{p(x | \theta)}}{p(x | \theta)}-\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)^{\mathrm{T}}\right] \\ &=\underset{p(x | \theta)}{\mathbb{E}}\left[\frac{\mathrm{H}_{p(x | \theta)}}{p(x | \theta)}\right]-\underset{p(x | \theta)}{\mathbb{E}}\left[\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)\left(\frac{\nabla p(x | \theta)}{p(x | \theta)}\right)^{\mathrm{T}}\right] \\ &=\int \frac{\mathrm{H}_{p(x | \theta)}}{p(x | \theta)} p(x | \theta) \mathrm{d} x-\underset{p(x | \theta)}{\mathbb{E}}\left[\nabla \log p(x | \theta) \nabla \log p(x | \theta)^{\mathrm{T}}\right] \\ &=\mathrm{H}_{\int p(x | \theta) \mathrm{d} x}-\mathrm{F} \\ &=\mathrm{H}_{1}-\mathrm{F} \\ &=-\mathrm{F} \end{aligned}
$$
所以我们可以得到$F=-\mathbb{E}_{P(x | \theta)}\left[\mathrm{H}_{\log p(x | \theta)}\right]$ ，而这里我们也可以看出$F$ 也可以作为log likelihood function的对curvature的度量

它经常用来衡量样本信息量，是一个非常有价值的指标。用来估计梯度时，可以松弛对代价函数的要求，不必使得代价函数可微，因此可以用来优化很多不可导的目标问题甚至是黑箱问题。







