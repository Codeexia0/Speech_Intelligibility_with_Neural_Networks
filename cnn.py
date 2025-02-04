from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / sigmoid for output
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=2), # 1 input channel, 16 output channels, 3x3 kernel size, 1 stride
            nn.ReLU(), # Activation function for non-linearity 
            nn.MaxPool2d(kernel_size=2) # downsampling the image representation by 2x2 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        

        
        self.flatten = nn.Flatten()  # Flatten the output of conv layers
        self.linear = nn.Linear(2800, 1)
        # self.linear = nn.Sequential(
            # nn.Linear(2800, 1), # 16*4*4
            # nn.ReLU(), # Activation function for non-linearity
            # nn.Linear(700, 175), 
            # nn.ReLU(),
            # nn.Linear(175, 25),  # Output layer
            # nn.ReLU(),
            # nn.Linear(25, 1)  # Output layer
        # )
            
            
    
        self.sigmoid = nn.Sigmoid()  # Sigmoid for output in range [0, 1]

    def forward(self, input_data):
        # print("shape of input", input_data.shape)
        x = self.conv1(input_data)
        # print("shape after conv1", x.shape)
        x = self.conv2(x)
        # print("shape after conv2", x.shape)
        x = self.conv3(x)
        # print("shape after conv3", x.shape)
        x = self.conv4(x)
        # print("shape after conv4", x.shape)
        x = self.flatten(x)
        # print("shape after flatten", x.shape)
        logits = self.linear(x)
        # print("shape after linear", logits.shape)
        predictions = self.sigmoid(logits)  # Apply Sigmoid to logits for regression
        # print("shape after sigmoid", predictions.shape)
        return predictions
        

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (2, 64, 537))  # Check model summary to confirm layer shapes and parameters