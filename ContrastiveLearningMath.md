### LogSumExp

---

Suppose **LogSumExp**(LSE) defined as:
$$
LSE(x_1, x_2, ..., x_n)=\log(\exp(x_1)+\exp(x_2)+...+\exp(x_n))
$$

- Property 1

$$
\max\{x_1,...,x_n\}\leq LSE(x_1,...,x_n)\leq \max\{x_1,...,x_n\} + \log(n)
$$

- $Proof$ 

	Let $x_m=\max \{x_1, ..., x_n\}$ . Then we have:
	$$
	\exp(x_m)\leq \sum_{i=1}^n\exp(x_i) \leq n \exp(x_m)
	$$
	 Applying the logarithm to the inequality:
	$$
	x_m \leq \log\sum_{i=1}^n\exp(x_i) \leq x_m + \log(n)
	$$

- Property 2

	In addition, we can scale the function to make the bounds tighter as below.

	For any $t>0$:

$$
\max\{x_1,...,x_n\} <  \frac{1}{t} LSE(tx_1,...,tx_n)\leq \max\{x_1,...,x_n\} + \frac{\log(n)}{t}
$$

- $Proof$

	Replace $x_i$ with $tx_i$ in the inequalities in Property 1. Then we have:
	$$
	\max\{tx_1,...,tx_n\}< LSE(tx_1,...,tx_n)\leq \max\{tx_1,...,tx_n\} + \log(n) \\
	t\max\{x_1,...,x_n\}< LSE(tx_1,...,tx_n)\leq t\max\{x_1,...,x_n\} + \log(n) \\
	$$
	Dividing by $t$ to the inequality:

$$
\max\{x_1,...,x_n\} <  \frac{1}{t} LSE(tx_1,...,tx_n)\leq \max\{x_1,...,x_n\} + \frac{\log(n)}{t}
$$

### Limit on Contrastive Learning Loss about $\tau$

---

**[Understanding the Behavior of Contrastive Loss. CVPR2021]**

Given an unlabeled training set $X=\{x_1, x_2, ..., x_N\}$, suppose $s_{i,j}=f(x_i)^T\cdot g(x_j)$, so the contrastive loss is formulated as :
$$
\mathcal{L}(x_i)=-\log\left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+\exp(s_{i,i}/\tau)}  \right]
$$


- Consider $\tau \rarr 0^{+}$ :
	$$
	\begin{align}
	&\lim_{\tau \rarr 0^{+}}-\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+ \exp(s_{i,i}/\tau)}  \right] \\ 
	=& \lim_{\tau \rarr 0^+}\frac{1}{\tau}\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\}
	\end{align}
	$$

	- $Proof$

	$$
	\begin{align}
	&\lim_{\tau \rarr 0^{+}}-\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+ \exp(s_{i,i}/\tau)}  \right] \\
	=& \lim_{\tau \rarr 0^{+}} \log \left[ 1 +{\sum_{k\neq i}\exp((s_{i,k}-s_{i,i})/\tau)}  \right] \\
	=& \lim_{\tau \rarr 0^{+}} \log \left[ \exp(0) +{\sum_{k\neq i}\exp((s_{i,k}-s_{i,i})/\tau)}  \right] \\
	=& \lim_{\tau \rarr 0^{+}} \log \left[ {\sum_{k}\exp((s_{i,k}-s_{i,i})/\tau)}  \right] \\
	
	\end{align}
	$$

Consider Property 2 of the LSE function ($\tau=\frac{1}{t}$), We have:
$$
\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\} <  \tau \cdot LSE(0, (s_{i,0}-s_{i,i})/\tau,...,(s_{i,k}-s_{i,i})/\tau))\leq \max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\} + \tau \cdot \log(n)
$$
Consider left function as a function about $\tau$ and apply limit on it:
$$
\lim_{\tau \rarr 0^{+}}\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\}=\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\}
$$
The same as the right function:
$$
\lim_{\tau \rarr 0^{+}}\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\} + \tau \cdot \log(n)=\max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\}
$$
According to **Squeeze Theorem** 
$$
\lim_{\tau \rarr 0^{+}} \tau \cdot\log \left[ {\sum_{k}\exp((s_{i,k}-s_{i,i})/\tau)}  \right]= \max\{0,(s_{i,0}-s_{i,i}),...,(s_{i,k}-s_{i,i})\}
$$

- Consider $\tau \rarr +\infty $ :
	$$
	\begin{align}
	&\lim_{\tau \rarr +\infty}-\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+ \exp(s_{i,i}/\tau)}  \right] \\ 
	=& \lim_{\tau \rarr +\infty}\frac{N-1}{N\tau}s_{i,i} + \frac{1}{N\tau} \sum_{k\neq i}s_{i,k} + \log N
	\end{align}
	$$
	

	- $Proof$

$$
\begin{align}
&\lim_{\tau \rarr + \infty}-\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+ \exp(s_{i,i}/\tau)}  \right] \\
=& \lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \log \sum_k \exp(s_{i,k}/\tau) \\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} +\log(N\cdot\left[\frac{1}{N} \sum_k \exp(s_{i,k}/\tau)-1+1  \right])\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \log(\left[\frac{1}{N}\sum_k \exp(s_{i,k}/\tau)-1+1  \right]) + \log N\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i}

\end{align}
$$

Consider the Taylor expansion of $\log(1+x)$  at $x_0=0$:
$$
\log(1+x)=\frac{1}{1+x_0}\cdot(x-x_0) + o(x)
$$
So Eq(24) should be:
$$
\begin{align}
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \log \left[ \left(\frac{1}{N}\sum_k \exp(s_{i,k}/\tau)-1\right)+1  \right]  + \log N\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\sum_k \exp(s_{i,k}/\tau)-1 + \log N

\end{align}
$$
Consider the Taylor expansion of $\exp(x) $ at  $x_0=0$ :
$$
\exp(x)=\exp(x_0) + \exp(x_0)(x-x_0) + o(x_0) \\
\exp(x)-1=x
$$
So Eq(27) should be:
$$
\begin{align}

=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\sum_k \exp(s_{i,k}/\tau)-1 + \log N \\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\left(\sum_k \exp(s_{i,k}/\tau)-N \right) + \log N\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\left(\sum_k [ \exp(s_{i,k}/\tau)-1] \right) + \log N\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\sum_ks_{i,k} + \log N\\
=&\lim_{\tau \rarr + \infty} -\frac{1}{\tau}s_{i,i} + \frac{1}{N}\sum_{k\neq i}s_{i,k} +\frac{1}{N}s_{i,i}+ \log N\\
=& \lim_{\tau \rarr +\infty}\frac{N-1}{N\tau}s_{i,i} + \frac{1}{N\tau} \sum_{k\neq i}s_{i,k} + \log N
\end{align}
$$

### Gradient Analysis

---

**[Understanding the Behavior of Contrastive Loss. CVPR2021]**

Given an unlabeled training set $X=\{x_1, x_2, ..., x_N\}$, suppose $s_{i,j}=f(x_i)^T\cdot g(x_j)$, so the contrastive loss is formulated as :
$$
\mathcal{L}(x_i)=-\log\left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+\exp(s_{i,i}/\tau)}  \right]
$$

- gradients with respect to the positive similarity $s_{i,i}$

$$
\begin{align}
\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,i}}
= & \frac{\partial}{\partial s_{i,i}}  -\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+\exp(s_{i,i}/\tau)} \right ] \\
= & \frac{\partial}{\partial s_{i,i}} \left[ \log \sum_k\exp(s_{i,k}/\tau) - \frac{s_{i,i}}{\tau} \right] \\
= & \frac{\partial}{\partial s_{i,i}}  \log \sum_k\exp(s_{i,k}/\tau) - \frac{1}{\tau} \\
= & \frac{1}{\sum_k \exp({s_{i,k}/\tau})} \cdot \exp(s_{i,i}/\tau)\cdot\frac{1}{\tau} - \frac{1}{\tau} \\
= & \frac{1}{\tau} \cdot \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_k\exp(s_{i,k}/\tau)} - 1 \right] \\
= & -\frac{1}{\tau} \cdot \left[ \frac{\sum_k\exp({s_{i,k}/\tau)} -  \exp(s_{i,i}/\tau)}{\sum_k\exp(s_{i,k}/\tau)} \right] \\

= & -\frac{1}{\tau} \cdot \frac{\sum_{k\neq i} \exp(s_{i,k}/\tau)}{\sum_k\exp(s_{i,k}/\tau)} \\
= & -\frac{1}{\tau} \cdot \sum_{k\neq i} \frac{\exp(s_{i,k}/\tau)}{\sum_j\exp(s_{i,j}/\tau)} \\

\end{align}
$$

- gradients with respect to the negative similarity $s_{i,k}$

$$
\begin{align}
\frac{\partial \mathcal{L}(x_i)}{\partial s_{i,j}}\bigg|_{i\neq j}
= & \frac{\partial}{\partial s_{i,j}}\bigg|_{i\neq j}  -\log \left[ \frac{\exp(s_{i,i}/\tau)}{\sum_{k\neq i}\exp(s_{i,k}/\tau)+\exp(s_{i,i}/\tau)} \right ] \\
= & \frac{\partial}{\partial s_{i,j}}\bigg|_{i\neq j} \left[ \log \sum_k\exp(s_{i,k}/\tau) - \frac{s_{i,i}}{\tau} \right] \\
= & \frac{\partial}{\partial s_{i,j}}\bigg|_{i\neq j}  \log \sum_k\exp(s_{i,k}/\tau) \\
= & \frac{1}{\sum_k \exp({s_{i,k}/\tau})} \cdot \exp(s_{i,j}/\tau)\cdot\frac{1}{\tau}  \\
= & \frac{1}{\tau} \cdot  \frac{\exp(s_{i,j}/\tau)}{\sum_k\exp(s_{i,k}/\tau)}  \\

\end{align}
$$

