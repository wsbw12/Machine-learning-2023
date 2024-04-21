import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogleNet, Inception


def test_data_process():  # 处理训练集和验证集
    test_data = FashionMNIST(root='./data', train=False,
                             # transforms.Resize(size=224)将图像调整为224x224像素的大小，
                             # transforms.ToTensor()将图像转换为张量格式。
                             transform=transforms.Compose([transforms.Resize(size=224),
                                                           transforms.ToTensor()]),
                             download=True)  # 如果数据集在指定路径下不存在，则下载FashionMNIST数据集。
    # batch_size 让测试数据一张一张的进行测试
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定训练所用的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将模型放入到训练的设备当中
    model = model.to(device)
    # 初始化参数
    test_corrects = 0.0
    test_num = 0
    # 模型只进行前向传播,不进行梯度计算
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将数据和标签放入到设备之中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程,输入为测试数据,输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确,则准确度test_corrects+1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)
    # 计算准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为: ", test_acc)


if __name__ == "__main__":
    # 加载模型
    model = GoogleNet(Inception)
    # 在"best_model.pth"加载模型的参数, model.load_state_dict 将模型序列化
    model.load_state_dict(torch.load("best_model.pth"))
    # 加载测试数据集
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    # test_model_process(model, test_dataloader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值: ", classes[result], "--------", "真实值: ", classes[label])
