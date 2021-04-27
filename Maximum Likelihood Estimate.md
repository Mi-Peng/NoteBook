## Maximum Likelihood Estimate
---
重要前提：iid

待续：
[详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解](https://blog.csdn.net/u011508640/article/details/72815981)
[从最大似然到EM算法浅解](https://blog.csdn.net/zouxy09/article/details/8537620)


&emsp;&emsp;多项式分布的MLE：

$$
\mathcal{L}(\theta)=\frac{n!}{m_1!m_2!...m_K!}\theta _1^{m_1}\theta _2^{m_2}...\theta _K^{m_K} \\
\begin{align}
\theta ^*&=arg\max _\theta \ log(\mathcal{L}(\theta)) \\
         &=arg\max _\theta\left[log(\frac{n!}{m_1!m_2!...m_K!}) + m_1\cdot log(\theta_1)+m_2\cdot log(\theta_2)  + ...+ m_K\cdot log(\theta_K)\right] \\
         &=arg\max_\theta \left[log(\frac{n!}{m_1!m_2!...m_K!}) +\sum_i^K m_ilog(\theta _i)\right] \\
         &=arg\max_\theta \sum_i^K m_ilog(\theta _i) \\
         & s.t. \sum_i^K\theta_i=1
\end{align}
$$

&emsp;&emsp;有条件求极值点

$$
\begin{cases}
		\max_{\theta} \sum_i^K m_ilog(\theta _i) \\
		s.t. \sum_i^K\theta_i=1
\end{cases}
$$

&emsp;&emsp;使用拉格朗日乘子法：

$$
\max_{\theta} \sum_i^K m_ilog(\theta _i) + \lambda (\sum_i^K\theta_i-1) \\
\begin{align}
取 \ \ \mathcal{L}&=\sum_i^K m_ilog(\theta _i) + \lambda (\sum_i^K\theta_i-1) 
\ \ 求导有 \\
&\begin{cases}
\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{m_1}{\theta_1}+\lambda\theta_1\\
\frac{\partial \mathcal{L}}{\partial \theta_2} = \frac{m_2}{\theta_2}+\lambda\theta_2\\
...\\
\frac{\partial \mathcal{L}}{\partial \theta_K} = \frac{m_K}{\theta_K}+\lambda\\
\frac{\partial \mathcal{L}}{\partial \lambda}=\sum_i^K\theta_i-1
\end{cases}
\\
\Rightarrow
&\begin{cases}
\forall \theta_i :\frac{\partial \mathcal{L}}{\partial \theta_i} = \frac{m_i}{\theta_i}+\lambda=0\\
\frac{\partial \mathcal{L}}{\partial \lambda}=\sum_i^K\theta_i-1=0
\end{cases}\\
\Rightarrow
&\begin{cases}
\forall \theta_i :\theta_i=-\frac{m_i}{\lambda}\\
\sum_i^K\theta_i=1
\end{cases}\\
\end{align}
$$

&emsp;&emsp;最后可得：

$$
\begin{align}
&\sum_i^K -\frac{m_i}{\lambda}=1 \\
&-\frac{\sum_i^K m_i}{\lambda}=1 \\
&-\frac{n}{\lambda} =1\\
&\lambda = -n
\end{align}
$$

&emsp;&emsp;代回公式(10)有：
$$
\theta_i = \frac{m_i}{n}
$$


### Ref
---
[极大似然估计求解多项式分布参数](https://blog.csdn.net/qy20115549/article/details/80232561)