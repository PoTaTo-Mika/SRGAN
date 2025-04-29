import math
import torch
from torch import nn
import torch.fft # 导入 torch.fft 模块

# 保留原始的 ResidualBlock 类定义，因为它将被用于新的 FCAModule 内部
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


# 定义新的 FCAModule，其内部使用 ResidualBlock
class FCAModuleWithResNet(nn.Module):
    def __init__(self, channels):
        super(FCAModuleWithResNet, self).__init__()
        self.channels = channels

        # 使用一个 ResidualBlock 替换原有的 MLP 来处理 pooled feature
        # 注意：ResidualBlock 通常处理 (B, C, H, W)，我们将 (B, C) reshape 成 (B, C, 1, 1) 输入给它
        self.attention_processor = ResidualBlock(channels) # 使用标准的 ResidualBlock

        # 最终用 Sigmoid 生成注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.size()

        # 1. 频率变换 (2D FFT 幅度谱)
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        x_freq_magnitude = torch.abs(x_freq)

        # 2. 在频率域表示上进行全局池化 (求和) -> (B, C)
        x_pooled = torch.sum(x_freq_magnitude, dim=(-2, -1))

        # 3. 将 pooled vector 形状调整为 (B, C, 1, 1)，作为 ResidualBlock 的输入
        x_pooled_reshaped = x_pooled.view(B, C, 1, 1)

        # 4. 将 reshape 后的特征输入到 ResidualBlock 进行处理
        attention_features_reshaped = self.attention_processor(x_pooled_reshaped)

        # 5. 将 ResidualBlock 的输出形状调整回 (B, C)
        attention_features = attention_features_reshaped.view(B, C)

        # 6. 应用 Sigmoid 获取通道注意力权重
        attention = self.sigmoid(attention_features) # Attention weights for each channel (B, C)

        # 7. 将注意力权重形状调整为 (B, C, 1, 1) 以便广播
        attention = attention.unsqueeze(-1).unsqueeze(-1)

        # 8. 将注意力权重应用于原始特征图
        output = x * attention

        return output


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

# 生成器类 (用于消融实验，将 ResidualBlock 替换为 FCAModuleWithResNet)
class GeneratorAblationFCA(nn.Module): # 为区分，重命名 Generator 类
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(GeneratorAblationFCA, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # 将原来的 ResidualBlock 替换为新的 FCAModuleWithResNet
        self.block2 = FCAModuleWithResNet(64)
        self.block3 = FCAModuleWithResNet(64)
        self.block4 = FCAModuleWithResNet(64)
        self.block5 = FCAModuleWithResNet(64)
        self.block6 = FCAModuleWithResNet(64)

        # block7 保持不变
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # block8 保持不变
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        # 前向传播结构与原 Generator 相同，只是 block2-block6 的类型变了
        block1 = self.block1(x)
        block2 = self.block2(block1) # 现在应用的是 FCAModuleWithResNet
        block3 = self.block3(block2) # 现在应用的是 FCAModuleWithResNet
        block4 = self.block4(block3) # 现在应用的是 FCAModuleWithResNet
        block5 = self.block5(block4) # 现在应用的是 FCAModuleWithResNet
        block6 = self.block6(block5) # 现在应用的是 FCAModuleWithResNet
        block7 = self.block7(block6)
        
        # 最终的跳跃连接也保持不变
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
        return torch.sigmoid(self.net(x).view(batch_size))