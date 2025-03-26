from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Block 1: from 2 channels to 32 channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # preserves spatial size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # downsample by 2
        )
        
        # Convolutional Block 2: from 32 channels to 64 channels
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Convolutional Block 3: from 64 channels to 128 channels
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Convolutional Block 4: from 128 channels to 256 channels
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Instead of flattening, use adaptive global average pooling.
        # This layer outputs a fixed-size feature map (1x1) per channel regardless of input size.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully-connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid maps output to [0, 1]
        )
    
    def forward(self, x):
        # Pass input through convolutional blocks
        x = self.conv_block1(x)
        print("shape after conv_block1: ", x.shape)
        x = self.conv_block2(x)
        print("shape after conv_block2: ", x.shape)
        x = self.conv_block3(x)
        print("shape after conv_block3: ", x.shape)
        x = self.conv_block4(x)
        print("shape after conv_block4: ", x.shape)
        # Global average pooling: output shape [B, 256, 1, 1]
        x = self.global_avg_pool(x)
        print("shape after global_avg_pool: ", x.shape)
        # Flatten the tensor: shape becomes [B, 256]
        x = x.view(x.size(0), -1) # Flatten the tensor: shape becomes [B, 256]
        print("shape after view: ", x.shape)
        # Fully-connected layers
        x = self.fc(x)
        print("shape after fc: ", x.shape)
        return x

if __name__ == "__main__":
    # Create an instance of the new CNN model
    cnn_v2 = CNNNetwork()
    # Print a summary of the model using torchsummary.
    # The expected input shape is (2, 64, 537) i.e. 2 channels, 64 frequency channels, 537 time steps.
    summary(cnn_v2.cuda(), (2, 64, 537))
