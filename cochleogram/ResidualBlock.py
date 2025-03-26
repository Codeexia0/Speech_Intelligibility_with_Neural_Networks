import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# Define your ResidualBlock if not already defined.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut: if the dimensions change, use a 1x1 convolution.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Full network using residual blocks.
class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Assume input shape: [B, 2, 64, 537] (2 channels, 64 frequency bins, 537 time steps)
        
        # First residual block: from 2 channels to 32 channels.
        self.block1 = ResidualBlock(in_channels=2, out_channels=32, stride=1)
        # Second block: from 32 channels to 64 channels, with spatial downsampling.
        self.block2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        # Third block: from 64 channels to 128 channels, with further downsampling.
        self.block3 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        
        # Global average pooling: outputs a fixed size [B, 128, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully-connected layers to map to one scalar output.
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid to map output to [0, 1]
        )
    
    def forward(self, x):
        # x: [B, 2, 64, 537]
        x = self.block1(x)   # Output shape: [B, 32, 64, 537] (if stride=1, spatial dims remain unchanged)
        x = self.block2(x)   # Downsampling: shape becomes [B, 64, H2, W2]
        x = self.block3(x)   # Further downsampling: shape becomes [B, 128, H3, W3]
        x = self.global_avg_pool(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)      # Flatten to [B, 128]
        x = self.fc(x)               # [B, 1]
        return x

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualCNN().to(DEVICE)
    # Check model summary (input shape: (2, 64, 537))
    summary(model, (2, 64, 537))
