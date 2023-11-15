import torch
from torchvision import transforms  # 数据的原始处理
from torchvision import datasets  # 加载数据集的
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 激活函数
import torch.optim as optim

# prepare dataset

batch_size = 64
# 把图片转换成模型里能训练的tensor也就是张量的格式
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 下载自带的数据集 其中一部分作为训练集 一部分作为测试集
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 设计训练模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义了第一个要用到的卷积层，因为图片输入通道为1，第一个参数就是1
        # 输出的通道为10，kernel_size是卷积核的大小，这里定义的是5x5的
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        # 定义第二个卷积层
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # 定义池化层
        self.pooling = torch.nn.MaxPool2d(2)
        # 定义一个线性层用于最后输出的分类
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)  # 这里面的0是x大小第1个参数，自动获取batch大小
        x = F.relu(self.pooling(self.conv1(x)))  # 输入x经过一个卷积层，之后经历一个池化层，最后用relu做激活
        x = F.relu(self.pooling(self.conv2(x)))  # 经过两次卷积，池化的处理
        x = x.view(batch_size, -1)  # 将最后的结果拉伸到一维
        x = self.fc(x)  # 经过线性层，确定他是0~9中某一个数的概率

        return x


model = Net()  # 对模型进行实例化

# 定义一个损失函数，来计算我们模型输出的值和标准值的差距
criterion = torch.nn.CrossEntropyLoss()
# 定义一个优化器 lr是设定的学习率
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 每次取一个样本
        inputs, target = data
        # 优化器中的梯度清零
        optimizer.zero_grad()
        # 正向计算一下输出结果
        outputs = model(inputs)
        # 计算此时的损失
        loss = criterion(outputs, target)
        # 反向计算此时的梯度
        loss.backward()
        # 进行更新参数和权重
        optimizer.step()
        # 将每一步的损失都加起来
        running_loss += loss.item()
        # 每迭代 300次输出一下数据
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 因为是测试验证所以不用计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # 取10个输出中概率最大的那个数作为输出
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            # 计算正确率
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
