# Why Classification Tasks Use CrossEntropy Instead of MSE

---

&emsp;&emsp;常见的解释有从sigmoid的导数角度出发的：

&emsp;&emsp;而我更倾向于另一种解释：

&emsp;&emsp;**MSE实际上是高斯分布的极大似然，CrossEntropy是多项式分布的极大似然，分类问题当然是多项式分布。**

&emsp;&emsp;多项式分布是二项分布的推广，在多项式分布之前，先来介绍二项分布与伯努利分布。

&emsp;&emsp; **伯努利分布(Bernoulli)**

&emsp;&emsp;伯努利分布是关于布尔变量 $x \in \{0,1\}$ 的概率分布，其连续参数 $p \in [0,1]$ 表示变量 $x=1$ 的概率。伯努利概率分布函数写作：

$$
P(x|p)=p^x \ (1-p)^{1-x}
$$

&emsp;&emsp;直观理解即扔一次硬币，硬币要么正面要么背面。

&emsp;&emsp;**二项分布(Binomial)**

&emsp;&emsp;二项分布即重复 $n$ 次独立的伯努利试验。在 $n$ 次独立重复的伯努利试验中，假设每次试验中事件 $A$ 发生的概率为 $p$ ，用 $X$ 表示 $n$ 重伯努利试验中事件 $A$ 发生的次数，则 $X$ 取值范围为 $0, 1, ..., n$ 。且对每一个$k(0 \leq k \leq n)$ ，事件{$X=k$}表示“$n$次试验中事件$A$恰好发生$k$次” 。随机变量$X$的离散概率分布即为二项分布。

&emsp;&emsp;一般地，如果随机变量 $X$ 服从参数为 $n$ 和 $p$ 的二项分布，我们记为 $X \sim B(n,p)$ 或 $X \sim b(n,p)$。 其离散概率分布为：

$$
\begin{align}
P(X=k|n,p)=&C_n^k\ p^k\ (1-p)^{n-k} \\
P(X=k|n,p)=&\left(^n_k  \right)p^k\ (1-p)^{n-k}
\end{align}
$$

&emsp;&emsp;直观理解即一枚硬币扔 $n$ 次，扔出正面的概率为 $p$ ，其中 $k$ 次正面的概率。

&emsp;&emsp;**多项分布(Multinomial Distribution)**

&emsp;&emsp;多项式分布是二项分布的推广，二项分布可以理解为抛一枚正面朝上概率为 $p$ 的硬币 $n$ 次，其中正面朝上 $k$ 次的概率分布。可以理解为多项式改成抛掷骰子。

$$
p(X_1=m_1,X_2=m_2,...,X_k=m_K|\theta _1, \theta _2, ..., \theta _K, n)=\frac{n!}{m_1!m_2!...m_K!}\theta _1^{m_1}\theta _2^{m_2}...\theta _K^{m_K}
$$


$$
\sum_{k=1}^K\theta _k = 1
$$

$$
\sum_{k=1}^Km_k = n
$$

&emsp;&emsp;多项式分布描述的是，一个事件有 $k$ 种结果，$\{x_1,x_2,...,x_k\}$，对应事件发生的概率为$\{\theta _1,\theta _2,...,\theta _k\}$，试验 $n$ 次，每种结果发生了$\{m_1,m_2,...,m_k\}$次。为了更直观，假设我们有一个灌铅骰子，我们定义随机变量 $X$ 为这个骰子朝上的点数，则 $X$ 可取 $1，2，3，4，5，6$ 。而概率依次为为$P(X=1)= \theta _1,P(X=2)= \theta _2,P(X=3)= \theta _3,P(X=4)= \theta _4,P(X=5)= \theta _5,P(X=6)= \theta _6$

，我们重复抛掷这个骰子 $n$ 次，结果一点朝上共$m_1$次，二点朝上共$m_2$次，三点朝上$m_3$次，....的概率即为公式$4$ 描述的事情。

> &emsp;&emsp;在理解该公式之前，我们了解一下“多项式”：
> $$
> (x_1+x_2+...+x_k)^n=\sum \frac{n!}{r_1!r_2!...r_k!}x_1^{r_1}x_2^{r_2}...x_k^{r_k}
> $$
>
> $$
> \begin{align}
> (x_1+x_2+...+x_K)^n&=\underbrace {(x_1+x_2+...+x_K)...(x_1+x_2+...+x_K)} _{n} \\
>                    &=\sum_{r_1,r_2,...,r_K}^{n} C_n^{r_1}C_{n-r_1}^{r_2}C_{n-r_1-r_2}^{r_3}...x_1^{r_1}x_2^{r_2}...x_K^{r_K} \\
>                    &=\sum_{r_1,r_2,...,r_K}^{n}  \left( \frac{n!}{(n-r_1)!} \frac{1}{r_1!}\right) \cdot \left( \frac{(n-r_1)!}{(n-r_1-r_2)!}\frac{1}{r_2!}\right) \cdot...\cdot \left( \frac{{(n-r_1-..-r_{k-1})}!}{(n-r_1-..-r_{K-1}-r_K)!}\frac{1}{r_K!} \right) \cdot x_1^{r_1}x_2^{r_2}...x_K^{r_K}   \\
>                    &=\sum_{r_1,r_2,...,r_K}^{n}  \frac{1}{r_1!} \frac{1}{r_2!} \cdot \cdot \cdot \frac{1}{r_K!} \cdot \frac{n!}{(n-r_1-..-r_{K-1}-r_K)!}\cdot x_1^{r_1}x_2^{r_2}...x_K^{r_K}  \\
>                    & \because \sum_i^K r_i = n \\
>                原式&=\sum_{r_1,r_2,...,r_K}^{n}  \frac{1}{r_1!} \frac{1}{r_2!} \cdot \cdot \cdot \frac{1}{r_K!} \cdot \frac{n!}{0!}\cdot x_1^{r_1}x_2^{r_2}...x_K^{r_K}  \\
>                    &=\sum_{r_1,r_2,...,r_K}^{n} \frac{n!}{r_1!r_2!...r_K!}x^{r_1}x^{r_2}...x^{r_K}
> \end{align}
> $$
>
> &emsp;&emsp;可以看出多项式分布的“多项式”名字的来源。把上面多项式中的$x_i$ 换成$\theta_i$，就可以发现多项式分布的离散概率分布即多项式展开式中的一个子项。多项式展开式的子项，描述多项式相乘时，每一次只能取括号种的一个项，一共取$n$次，前面的系数即有多少种取法。

&emsp;&emsp;我们回顾交叉熵：
$$
\mathcal{Loss}=\sum_i^K -y_ilog(p_ {\theta _i}(x))
$$

&emsp;&emsp;我们模型所学习的即找一个网络参数$\theta^*$ 使模型交叉熵最小：

$$
\begin{align}
\theta^* &= \arg\min_{\theta} \mathcal{Loss} \\
       &= \arg \min_{\theta} \frac{1}{|B|}\sum_{x\in B}\sum_i^K -y_ilog(p(x|\theta))
\end{align}
$$


&emsp;&emsp;其中$y_i$ 为$label$，$p_ {\theta _i}(x)$为模型输出的所有类别概率值。那么这是不是多项式分布的最大似然估计？我们考虑多项式分布 **$\theta$** 的对数似然函数:
$$
\mathcal{L}(\theta|X)=\frac{n!}{m_1!m_2!...m_K!}\theta _1^{m_1}\theta _2^{m_2}...\theta _K^{m_K} \\

\begin{align}
log(\mathcal{L}(\theta|X))&=log(\frac{n!}{m_1!m_2!...m_K!}) + m_1\times log(\theta_1)+m_2\times log(\theta_2)  + ...+ m_K\times log(\theta_K) \\
&=log(\frac{n!}{m_1!m_2!...m_K!}) +\sum_i^K m_ilog(\theta _i)
\end{align}
$$

&emsp;&emsp;我们的$label$是 one-hot向量，即，认为只有$label$对应的事件发生概率为1，其余事件概率为0；只进行一次试验（$n=1$ 的情况下 只有$m_{label}=1$ ，其余为0）我们希望我们的分布输出能够尽可能与这种情况相符，也即：

$$
\begin{align}
\theta^* &= \arg\max_{\theta} log(\mathcal{L}(\theta|X)) \\
         &= \arg \max_{\theta}log(\frac{n!}{m_1!m_2!...m_K!}) +\sum_i^K m_ilog(\theta _i) \\
         &= \arg \max_{\theta}\sum_i^K m_ilog(\theta _i) \\
         &= \arg \max_{\theta} m_j \times log(\theta_j)
\end{align}
$$


&emsp;&emsp;已知试验结果为"$label$"项对应的事件下对这个多项式分布做极大似然估计，得出这个多项式分布的参数的过程即等价于$label$事件发生的概率为1，与分类任务使用交叉熵作为$\mathcal{Loss}$对模型参数进行训练做的是同样的过程。

&emsp;&emsp;那么MSE又发生了什么？

$$
p(y|x) \sim \mathcal{N}(\mu,\sigma) \\
p(y|x)=\frac{1}{\sqrt{(2\pi)\cdot\sigma^2}}\cdot exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}
$$

&emsp;&emsp;其对数似然函数有：
$$
log(\mathcal{L}(\mu,\sigma|X))=-\frac{1}{2}log2\pi - log(\sigma) -\frac{(x-\mu)^2}{2\sigma^2}\
$$

&emsp;&emsp;最后一项等价于MSE损失函数。

### Ref

---
* [最大似然估计，MSE Loss和高斯分布](https://zhuanlan.zhihu.com/p/258961357) 

* [从最大似然估计理解交叉熵Loss](https://zhuanlan.zhihu.com/p/145967829)