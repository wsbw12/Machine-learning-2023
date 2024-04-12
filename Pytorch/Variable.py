import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b
y.backward()  # 对所有需要梯度的变量进行求导
print(x.grad)
print(w.grad)
print(b.grad)

x = torch.rand(3)
x = Variable(x, requires_grad=True)
y = x * 2
# y是一个向量不能对向量直接写y.backward()
# 对向量的求导需要写入传入的参数声明  y.backward(torch.FloatTensor([1, 0.1, 0.01]))
# 这样得到的梯度就是它们原本的梯度乘以[1,0.1,0.01]
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)
