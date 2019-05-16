# LeakGAN

$s_t = (x_1,...,x_i,...,x_t)$ 这里的$x_i$代表的是vocabulary中的一个token，$G_\theta$ 是生成网络，也是stochastic policy，生成$x_{t+1}$ 。当整句话$s_T$ 生成出来后，判别器$D_\phi(s_T)$ 会给$G_\theta$提供一个监督信号。

$D_{\phi}(s)=\operatorname{sigmoid}\left(\phi_{l}^{\top} \mathcal{F}\left(s ; \phi_{f}\right)\right)=\operatorname{sigmoid}\left(\phi_{l}^{\top} f\right)$ 

$D_\phi$ 的最后一层是一个Sigmoid的分类器，之前是feature extractor（这里用的是2D CNN+maxpooling 按照不同filter进行卷积，最后将不同尺度的卷积结果concate，而这之后代码中还用到了highway network的操作，**可以把这里改为resNet结构 或者用LSTM替换CNN**)

在$G_\theta$ 中，从$D_\phi$ 中得到的feature $f_t$ 作为输入，通过LSTM输出$g_t$ 。

$G_\theta$ 中，有manager和worker两个模块，hidden state初始化为0。manager模块的LSTM得到的$g_t$ 进行normalization

c=4

we perform one epoch of supervised learning for G after 15 epochs of adversarial training