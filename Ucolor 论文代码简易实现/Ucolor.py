import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiColorSpaceEncoder(nn.Module):
    def __init__(self, in_channels):
        super(MultiColorSpaceEncoder, self).__init__()

        # 创建RGB、HSV和LAB编码器
        self.rgb_encoder = self.build_encoder(in_channels)
        self.hsv_encoder = self.build_encoder(in_channels)
        self.lab_encoder = self.build_encoder(in_channels)

    def build_encoder(self, in_channels):
        # 构建编码器结构
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # 3x3卷积，64个输出通道
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 3x3卷积，128个输出通道
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化层
        )
        return encoder

    def forward(self, x):
        # 将输入特征图分别传递给RGB、HSV和LAB编码器
        rgb_features = self.rgb_encoder(x)
        hsv_features = self.hsv_encoder(x)
        lab_features = self.lab_encoder(x)
        return rgb_features, hsv_features, lab_features


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 初始化通道注意力模块的参数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，用于提取全局特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化层，也用于提取全局特征
        # 使用全连接层来学习通道的权重
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),  # 特征压缩
            nn.ReLU(),  # 激活函数
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),  # 特征恢复
            nn.Sigmoid()  # Sigmoid函数，用于输出通道的注意力权重
        )

    def forward(self, x):
        # 计算全局平均池化和全局最大池化
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        # 将两个池化结果连接起来
        out = avg_out + max_out
        # 通过全连接层计算通道权重
        out = self.fc(out)
        # 将通道权重乘以输入特征图
        out = x * out
        return out


class MediumTransmissionGuidanceModule(nn.Module):
    def __init__(self, in_channels):
        super(MediumTransmissionGuidanceModule, self).__init__()
        # 定义一个卷积层，用于处理输入特征图
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rmt_map):
        # 将介质透射图应用于输入特征图，实现对不同区域的加权
        out = torch.mul(x, rmt_map)
        # 通过卷积层处理加权后的特征图，以获得更好的表示能力
        out = self.conv(out)
        return out


class ResidualEnhancementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class Ucolor(nn.Module):
    def __init__(self, in_channels):
        super(Ucolor, self).__init__()

        # 创建多颜色空间编码器
        self.encoder = MultiColorSpaceEncoder(in_channels)

        # 创建RGB、HSV和LAB通道注意力模块
        self.cam_rgb = ChannelAttentionModule(128)
        self.cam_hsv = ChannelAttentionModule(128)
        self.cam_lab = ChannelAttentionModule(128)

        # 创建介质透射引导模块
        self.decoder = MediumTransmissionGuidanceModule(384)

        # 创建三个残差增强模块
        self.residual_enhancement1 = ResidualEnhancementModule(384, 384)
        self.residual_enhancement2 = ResidualEnhancementModule(384, 384)
        self.residual_enhancement3 = ResidualEnhancementModule(384, 384)

        # 创建最终输出的卷积层
        self.conv_out = nn.Conv2d(384, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入图像经过多颜色空间编码器编码
        rgb_features, hsv_features, lab_features = self.encoder(x)

        # RGB、HSV和LAB特征通过通道注意力模块增强
        rgb_features = self.cam_rgb(rgb_features)
        hsv_features = self.cam_hsv(hsv_features)
        lab_features = self.cam_lab(lab_features)

        # 将增强后的特征拼接在一起
        features = torch.cat((rgb_features, hsv_features, lab_features), dim=1)

        # 通过介质透射引导模块处理特征
        x_transmission = self.decoder(features)

        # 特征通过三个残差增强模块增强
        out = self.residual_enhancement1(x_transmission)
        out = self.residual_enhancement2(out)
        out = self.residual_enhancement3(out)

        # 最终特征经过卷积层输出
        out = self.conv_out(out)

        # 返回输出结果
        return out
