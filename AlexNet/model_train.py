import copy
import time
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet
from torch import nn
import pandas as pd


def train_val_data_process():  # 处理训练集和验证集
    train_data = FashionMNIST(root='./data', train=True,
                              # transforms.Resize(size=224)将图像调整为224x224像素的大小，
                              # transforms.ToTensor()将图像转换为张量格式。
                              transform=transforms.Compose([transforms.Resize(size=227),
                                                            transforms.ToTensor()]),
                              download=True)  # 如果数据集在指定路径下不存在，则下载FashionMNIST数据集。
    # 划分训练集和验证集
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义优化器  Adam 优化后的梯度下降法
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义交叉熵损失函数来更新参数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备当中
    model = model.to(device)
    # 复制当前的模型参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化参数
    best_acc = 0.0
    train_loss_all = []  # 设置列表保存训练集的loss值
    val_loss_all = []  # 设置列表保存验证集的loss值
    train_acc_all = []  # 设置列表保存训练集的精确值
    val_acc_all = []  # 设置列表保存验证集的精确值
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化训练集的损失和准确度
        train_loss = 0.0
        train_corrects = 0
        # 初始化验证集的损失和准确度
        val_loss = 0.0
        val_corrects = 0
        # 训练集的样本数量
        train_num = 0
        # 验证集的样本数量
        val_num = 0

        # 对每一个min_batch进行训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)  # 将特征放入训练设备中
            b_y = b_y.to(device)  # 将标签放入训练设备中
            # 设置模型为训练模式
            model.train()
            # 前向传播过程，输入一个batch，输出一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的下标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch损失
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 进行反向传播
            loss.backward()
            # 根据反向传播的梯度信息来更新网络的参数,以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确,则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # 验证集的过程
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)  # 将特征放入验证设备中
            b_y = b_y.to(device)  # 将标签放入验证设备中
            model.eval()  # 设置模型为评估模式
            output = model(b_x)  # 得出模型的预测结果
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch对应的损失
            loss = criterion(output, b_y)
            # 对损失值进行累加
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 用于验证集的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度的权重参数
        if val_acc_all[-1] > best_acc:
            # 保存当前最高的准确度
            best_acc = val_acc_all[-1]
            # 保存当前模型的参数
            best_model_wts = copy.deepcopy(model.state_dict())
        # 该轮次训练一共花费的时间
        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数
    # model.load_state_dict(best_model_wts)

    torch.save(best_model_wts, '/Users/wangsibo/PycharmProjects/AlexNet/best_model.pth')
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # 1行两列的第一张图

    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)  # 1行两列的第2张图
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="acc_loss")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="acc_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.show()


if __name__ == "__main__":
    # 将模型进行实例化
    AlexNet = AlexNet()
    # 加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    # 训练
    train_process = train_model_process(AlexNet, train_dataloader, val_dataloader, num_epochs=20)
    # 画图
    matplot_acc_loss(train_process)
