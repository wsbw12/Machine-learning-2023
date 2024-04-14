import torch
from torch.autograd import Variable
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt


def make_features(x):
    # [x,x^2,x^3,...1]
    # 将张量x进行维度扩展 原来tensor大小是3变为(3,1)
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(w_target) + b_target


def get_batch(batch_size=32):
    # 每次取batch_size这么多个数据点,然后将其转化为矩阵的形式
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if (epoch + 1) % 20 == 0:
        print('Epoch[{}],loss:{:.6f}'.format(epoch + 1, loss.data))
    if print_loss < 1e-3:
        break
