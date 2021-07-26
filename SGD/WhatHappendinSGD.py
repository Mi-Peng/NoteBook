import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.data = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
        self.label = 0.8 * self.data + 1.2

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


model = MyModel()
data = MyData()
# learning rate
LR = 0.1
# weight decay
WD = 0.5
# momentum
M = 0.9

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=M, weight_decay=WD, dampening=0, nesterov=False)
w = model.layer.weight.data.item()
b = model.layer.bias.data.item()

# Official Pytorch Implement
for _ in range(5):
    inputs, target = data[_]
    output = model(inputs)
    loss = F.mse_loss(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(model.layer.weight.data, model.layer.bias.data)

# Our Code
w_grad_last = 0.
b_grad_last = 0.
for _ in range(5):
    x, y = data[_]
    w_grad = 2 * (w * x + b - y) * x
    b_grad = 2 * (w * x + b - y)
    if WD != 0:
        w_grad = w_grad + WD * w
        b_grad = b_grad + WD * b
    if M != 0:
        w_grad = M * w_grad_last + w_grad
        b_grad = M * b_grad_last + b_grad
    new_w = w - LR * w_grad
    new_b = b - LR * b_grad

    w = new_w
    b = new_b
    w_grad_last = w_grad
    b_grad_last = b_grad
print(w, b)


