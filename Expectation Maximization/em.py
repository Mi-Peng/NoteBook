import numpy as np
import matplotlib.pyplot as plt
num = 100

mu1 = 20
sigma1 = 5
d1 = np.random.normal(mu1, sigma1, num)

mu2 = 50
sigma2 = 8
d2 = np.random.normal(mu2, sigma2, num)

data = np.append(d1, d2)
np.random.shuffle(data)

#EM:
# init:
f_mu1 = 20
f_sigma1 = 4
f_mu2 = 25
f_sigma2 = 5


def gaussian(mu, sigma, x):
    p = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(- (x - mu) ** 2/(2 * sigma * sigma))
    return p


epochs = 1000
for epoch in range(epochs):
    data_1 = np.array([])
    data_2 = np.array([])
    for adata in data:
        P1 = gaussian(f_mu1, f_sigma1, adata)
        P2 = gaussian(f_mu2, f_sigma2, adata)
        if P1 > P2:
            data_1 = np.append(data_1, adata)
        else:
            data_2 = np.append(data_2, adata)
    # MLE
    f_mu1 = np.sum(data_1) / data_1.shape[0]
    f_sigma1 = np.sqrt(np.sum((data_1 - f_mu1)**2) / data_1.shape[0])

    f_mu2 = np.sum(data_2) / data_2.shape[0]
    f_sigma2 = np.sqrt(np.sum((data_2 - f_mu2) ** 2) / data_2.shape[0])
    print('epoch:', epoch)
print(f_mu1, f_sigma1)
print(f_mu2, f_sigma2)