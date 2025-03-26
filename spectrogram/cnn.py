from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()

        # Convolutional Layers with Batch Normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive Pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(16, 1),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = self.conv1(input_data)
        print("shape after conv1: ", x.shape)
        x = self.conv2(x)
        print("shape after conv2: ", x.shape)
        x = self.conv3(x)
        print("shape after conv3: ", x.shape)   
        x = self.conv4(x)
        print("shape after conv4: ", x.shape)
        x = self.adaptive_pool(x)
        print("shape after adaptive_pool: ", x.shape)
        x = self.flatten(x)
        print("shape after flatten: ", x.shape)
        predictions = self.fc(x)  # No sigmoid; regression outputs raw values
        predictions = self.sigmoid(predictions)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (2, 64, 169))  # Check model summary for layer shapes and parameter counts