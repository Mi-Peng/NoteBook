**GD**

**SGD**

**MBGD**



Consider the iterative equation of **Gradient Descent(GD)** is : ($\eta$​​​ is the learning rate, $N$​ is the number of the train dataset)
$$
\theta_{t+1}=\theta_t - \eta \cdot \nabla_{\theta_t} \mathcal{L}
$$

$$
\nabla_{\theta_t}\mathcal{L}=\frac{1}{N}\sum_{i=1}^N\mathcal{L}\left(f_{\theta_t}(x),\text{GroundTruth}\right)
$$

Note that the gradient descent use **all** train dataset to compute the gradient in each iteration.



Consider the iterative equation of **Mini-Batch Gradient Descent(MBGD)** is:($\eta$​ is the learning rate, $B$​​ is batch size, which is the part of the train dataset)
$$
\theta_{t+1}=\theta_t - \eta \cdot \nabla_{\theta_t} \mathcal{L}
$$

$$
\nabla_{\theta_t}\mathcal{L}=\frac{1}{B}\sum_{i=1}^B\mathcal{L}\left(f_{\theta_t}(x),\text{GroundTruth}\right)
$$

Note that **MBGD** bring in the noise. 



Consider the iterative equation of **Stochastic Gradient Descent(SGD)** is:($\eta$ is the learning rate, $B$ is batch size, which is the part of the train dataset)
$$
\theta_{t+1}=\theta_t - \eta \cdot \nabla_{\theta_t} \mathcal{L}
$$

$$
\nabla_{\theta_t}\mathcal{L}=\mathcal{L}\left(f_{\theta_t}(x),\text{GroundTruth}\right)
$$

Note that the stochastic gradient descent use **one** train dataset to compute the gradient in each iteration and also bring in the large noise.



Actually, the definition of GD,MBGD and SGD is not very clear. Actually in **Pytorch**, the **torch.optim.SGD** is the mini-batch gradient descent. 

```python
torch.optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
'''
lr(float): learning rate
momentum(float): momentum factor
weight_decay(float): weight decay (L2 penalty)
dampening(float): dampening for momentum
nesterov(bool): 
'''
```

