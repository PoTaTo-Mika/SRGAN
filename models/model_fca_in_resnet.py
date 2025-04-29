import math
import torch
from torch import nn
import torch.fft # 导入 torch.fft 模块

# 定义频率通道注意力模块
class FCAModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FCAModule, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 使用全连接层从频率域特征生成注意力权重
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)

        # 1. 频率变换 (2D FFT 幅度谱)
        # rfft2 用于实数输入，返回复数结果，形状为 (B, C, H, W//2 + 1)
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        # 获取幅度谱
        x_freq_magnitude = torch.abs(x_freq)

        # 2. 在频率域表示上进行全局池化
        # 在频率维度 (H, W//2 + 1) 上求和作为通道描述符
        # 也可以尝试平均池化，这里选择求和
        x_pooled = torch.sum(x_freq_magnitude, dim=(-2, -1))

        # 3. 通过全连接层生成通道注意力权重
        attention = self.fc1(x_pooled)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention) # 每个通道的注意力权重 (B, C)

        # 调整注意力权重的形状以便广播 (B, C, 1, 1)
        attention = attention.unsqueeze(-1).unsqueeze(-1)

        # 4. 将注意力权重应用于原始特征图
        output = x * attention

        return output


# 修改 ResidualBlock 类，集成 FCAModule
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 在第二个 BN 之后添加 FCAModule
        self.fca = FCAModule(channels) # 使用新的 FCAModule

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # 应用 FCAModule 到残差路径上
        residual = self.fca(residual)

        # 将加权的残差加到输入上
        return x + residual


# UpsampleBLock 类 (保持不变)
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# Generator 类 (使用修改后的 ResidualBlock)
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # 这里将使用上面定义的修改后的 ResidualBlock
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


# Discriminator 类 (保持不变)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 鉴别器的输出通常是单个值，表示是真实图片的概率
        return torch.sigmoid(self.net(x).view(batch_size))