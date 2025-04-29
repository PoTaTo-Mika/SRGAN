import math
import torch
from torch import nn
import torch.fft

# Define Frequency Channel Attention Module with explainable frequency filtering
class ExplainableFCAModule(nn.Module):
    def __init__(self, channels, reduction=16, freq_dim_h=64, freq_dim_w_half=33): # Example frequency dimensions, adjust based on expected input size and rfft2 output
        super(ExplainableFCAModule, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.freq_dim_h = freq_dim_h
        self.freq_dim_w_half = freq_dim_w_half

        # Learnable frequency weight mask
        # Initialize with ones or a small random value; ones means no initial filtering
        self.frequency_weights = nn.Parameter(torch.ones(channels, freq_dim_h, freq_dim_w_half))

        # Use full connection layers to generate attention weights from weighted frequency features
        # Input size is flattened weighted frequency features
        self.fc1 = nn.Linear(channels * freq_dim_h * freq_dim_w_half, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)

        # 1. Frequency Transformation (2D FFT Magnitude Spectrum)
        # rfft2 for real input, returns complex result shape (B, C, H, W//2 + 1)
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        x_freq_magnitude = torch.abs(x_freq) # Shape: (B, C, H, W//2 + 1)

        # Ensure the frequency weights match the current input size if dimensions change
        # This is a basic check; for variable input sizes, more sophisticated handling is needed (e.g., adaptive pooling)
        if x_freq_magnitude.shape[-2:] != (self.freq_dim_h, self.freq_dim_w_half):
             # For variable input sizes, you might need to resize self.frequency_weights
             # or use adaptive pooling on the frequency magnitude spectrum before weighting.
             # For simplicity in this example, we'll assume a fixed input size for which freq_dim_h and freq_dim_w_half are set.
             # In a real-world scenario with variable sizes, consider adaptive methods.
             # As a temporary measure, let's just resize the magnitude to match the weight size if needed and warn.
             import warnings
             warnings.warn(f"Frequency magnitude size {x_freq_magnitude.shape[-2:]} does not match expected {self.freq_dim_h, self.freq_dim_w_half}. Resizing magnitude spectrum for weighting. Consider using adaptive pooling or setting freq_dim_h/w_half correctly.")
             # Simple resizing for demonstration - this might not be ideal depending on the application
             x_freq_magnitude_resized = torch.nn.functional.interpolate(x_freq_magnitude, size=(self.freq_dim_h, self.freq_dim_w_half), mode='bilinear', align_corners=False)
             x_freq_magnitude = x_freq_magnitude_resized


        # Apply learnable frequency weights
        # Expand weights to match batch size for element-wise multiplication
        frequency_weights_expanded = self.frequency_weights.unsqueeze(0).expand(x_freq_magnitude.size(0), -1, -1, -1)
        weighted_x_freq_magnitude = x_freq_magnitude * frequency_weights_expanded

        # Flatten the weighted frequency features for the FC layers
        batch_size, channels, h, w = weighted_x_freq_magnitude.size()
        # Shape: (B, C * H * W//2+1)
        flattened_weighted_freq = weighted_x_freq_magnitude.view(batch_size, -1)


        # 3. Generate Channel Attention Weights using FC layers on the flattened weighted frequency features
        attention = self.fc1(flattened_weighted_freq)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention) # Channel attention weights (B, C)

        # Reshape attention for broadcasting (B, C, 1, 1)
        attention = attention.unsqueeze(-1).unsqueeze(-1)

        # 4. Apply Attention Weights to Original Feature Map
        output = x * attention

        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Use the new ExplainableFCAModule
        # For a 4x super-resolution and 256x256 HR crop, LR input is 64x64.
        # Feature map size before rfft2 in FCAModule will be 64x64.
        # rfft2(64x64) -> magnitude spectrum size 64 x (64//2 + 1) = 64 x 33
        self.fca = ExplainableFCAModule(channels, freq_dim_h=64, freq_dim_w_half=33)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Apply ExplainableFCAModule to the residual path
        residual = self.fca(residual)

        # Add the weighted residual to the input
        return x + residual


# UpsampleBLock Class (kept the same as original model_fca_in_resnet.py)
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

# Generator Class (using the modified ResidualBlock)
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # Use the ResidualBlock containing ExplainableFCAModule
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


# Discriminator Class (kept the same as original model_fca_in_resnet.py)
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
        # The output of the discriminator is typically a single value representing the probability of the image being real
        return torch.sigmoid(self.net(x).view(batch_size))