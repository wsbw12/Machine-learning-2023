import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 以上都是约定俗成的写法
        self.ReLu = nn.ReLU()  # 定义激活函数
        # 输入图像为 227*227*1
        # 第一个卷积层输入为 227*227*1,卷积核为11*11*1*96;步幅为4,输出为55*55*96
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, stride=4, kernel_size=11)
        # 第二个为池化层,输入为55*55*96,池化感受野为3*3,stride=2,output为27*27*96
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第三个卷积层输入为 27*27*96,卷积核为5*5*96*256;步幅为1,padding=2,输出为27*27*256
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, stride=1, padding=2, kernel_size=5)
        # 第四个为池化层,输入为27*27*256,池化感受野为3*3,stride=2,output为13*13*256
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第五个卷积层输入为 13*13*256,卷积核为3*3*256*384;步幅为1,padding=1,输出为13*13*384
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, stride=1, padding=1, kernel_size=3)
        # 第六个卷积层输入为 13*13*384,卷积核为3*3*384*384;步幅为1,padding=1,输出为13*13*384
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, stride=1, padding=1, kernel_size=3)
        # 第七个卷积层输入为 13*13*384,卷积核为3*3*384*256;步幅为1,padding=1,输出为13*13*256
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, stride=1, padding=1, kernel_size=3)
        # 第八个为池化层,输入为13*13*256,池化感受野为3*3,stride=2,output为6*6*256
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()  # 定义一个平展层
        self.f1 = nn.Linear(6 * 6 * 256, 4096)  # 定义全连接层
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.ReLu(self.c1(x))
        x = self.s2(x)
        x = self.ReLu(self.c3(x))
        x = self.s4(x)
        x = self.ReLu(self.c5(x))
        x = self.ReLu(self.c6(x))
        x = self.ReLu(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)
        x = self.ReLu(self.f1(x))
        x = F.dropout(x, 0.5)  # 0.5的概率进行随机失活
        x = self.ReLu(self.f2(x))
        x = F.dropout(x, 0.5)  # 0.5的概率进行随机失活
        x = self.ReLu(self.f3(x))
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))
