# Why Classification Tasks Use CrossEntropy Instead of MSE
---

&emsp;&emsp;常见的解释有从sigmoid的导数角度出发的：

&emsp;&emsp;而我更倾向于：

&emsp;&emsp;MSE实际上是高斯分布的极大似然，CrossEntropy是多项式分布的极大似然，分类问题当然是多项式分布。

&emsp;&emsp;多项式分布是二项分布的推广，在多项式分布之前，先来介绍二项分布。

&emsp;&emsp;伯努利分布(Bernoulli)

&emsp;&emsp;伯努利分布是关于布尔变量 $x \in \{0,1\}$ 的概率分布，其连续参数 $p \in [0,1]$ 表示变量 $x=1$ 的概率。伯努利概率分布函数写作：

$$
P(x|p)=p^x \ (1-p)^{1-x}
$$

&emsp;&emsp;直观理解即扔一次硬币，硬币要么正面要么背面。

&emsp;&emsp;二项分布(Binomial)

&emsp;&emsp;二项分布即重复 $n$ 次独立的伯努利试验。在 $n$ 次独立重复的伯努利试验中，假设每次试验中事件 $A$ 发生的概率为 $p$ ，用 $X$ 表示 $n$ 重伯努利试验中事件 $A$ 发生的次数，则 $X$ 取值范围为 $0, 1, ..., n$ 。且对每一个$k(0 \leq k \leq n)$ ，事件{$X=k$}表示“$n$次试验中事件$A$恰好发生$k$次” 。随机变量$X$的离散概率分布即为二项分布。

&emsp;&emsp;一般地，如果随机变量 $X$ 服从参数为 $n$ 和 $p$ 的二项分布，我们记为 $X \sim B(n,p)$ 或 $X \sim b(n,p)$。 其离散概率分布为：

$$
\begin{align}
P(X=k)=&C_n^k\ p^k\ (1-p)^{n-k} \\
P(X=k)=&\left(^n_k  \right)p^k\ (1-p)^{n-k}
\end{align}
$$

&emsp;&emsp;直观理解即一枚硬币扔 $n$ 次，扔出正面的概率为 $p$ ，其中 $k$ 次正面的概率。

&emsp;&emsp;多项分布(Multinomial Distribution)

&emsp;&emsp;多项式分布是二项分布的推广，二项分布可以理解为抛一枚正面朝上概率为 $p$ 的硬币 $n$ 次，其中正面朝上 $k$ 次。可以理解为多项式改成抛掷骰子或者别的什么。