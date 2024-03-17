from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

train_data = FashionMNIST(root='./data', train=True,
                          # transforms.Resize(size=224)将图像调整为224x224像素的大小，
                          # transforms.ToTensor()将图像转换为张量格式。
                          transform=transforms.Compose([transforms.Resize(size=224),
                                                        transforms.ToTensor()]),
                          download=True)  # 如果数据集在指定路径下不存在，则下载FashionMNIST数据集。
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,  # 指定每个批次中包含的样本数量。这里设置为64，表示每次加载64个样本
                               shuffle=True,  # 表是要对训练数据进行打乱
                               num_workers=0)  # 指定用于数据加载的子进程数目。这里设置为0，表示数据加载将在主进程中进行。
# 通常在数据加载较慢时，可以增加num_workers来加快数据加载速度。

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第一维,并转换为numpy()数组
batch_y = b_y.numpy()  # 将张量转换为Numpy数组
class_label = train_data.classes  # 训练集的标签
print(class_label)
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 可视化Batch图像
plt.figure(figsize=(12, 5))
for li in np.arange(len(batch_y)):
    plt.subplot(4, 16, li + 1)
    plt.imshow(batch_x[li, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[li]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
