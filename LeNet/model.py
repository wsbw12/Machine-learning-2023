import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    # 初始化参数
    def __init__(self):
        super(LeNet, self).__init__()
        # Input为28*28*1,卷积核为5*5*1*6,stride(步幅)为1,padding(填充)为2,输出为28*28*6
        # 第一个卷积层, 输入通道in_channels为1,输出通道out_channels为6,卷积核的大小kernel_size为5,填充层为2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()  # 定义激活函数
        # 平均池化层 Input为28*28*6,池化感受野为2*2,stride(步幅)为2,输出为14*14*6
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 定义池化层
        # 第二个卷积层,Input为14*14*6,卷积核为5*5*6*16,stride(步幅)为1,padding(填充)为0,输出为10*10*16
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 平均池化层 Input为10*10*16,池化感受野为2*2,stride(步幅)为2,输出为5*5*16
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()  # 平展层
        self.f5 = nn.Linear(5 * 5 * 16, 120)  # 线性全连接层 输入5*5*16 输出是120
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    # 前向传播的函数
    def forward(self, x):
        x = self.sig(self.c1(x))  # 将数据进入第一个卷积层之后进入sigmoid函数
        x = self.s2(x)  # 进入池化层
        x = self.sig(self.c3(x))  # 将数据进入第二个卷积层之后进入sigmoid函数
        x = self.s4(x)  # 进入池化层
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))  # 输入数据是1*28*28
