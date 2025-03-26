import torch
from torch import nn
from torchsummary import summary

class CNN_RNN_Hierarchical(nn.Module): # CNN_RNN_Hierarchical: Hierarchical Convolutional Recurrent Neural Network
    def __init__(self, freq_bins=64, rnn_hidden_size=128):
        super().__init__()
        # Local feature extraction with CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # input: 2 channels, output: 32 channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),  # Reduce frequency dimension by 2 (from 64 to 32)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))   # Further reduce frequency dimension by 2 (from 32 to 16)
        )
        
        # After CNN, the feature map is [B, 64, freq_bins/4, T]. For freq_bins=64, freq_bins/4=16.
        # We'll treat the time dimension T as the sequence length.
        # Flatten spatial dimensions except for the time dimension.
        self.flattened_feature_size = 64 * 16
        
        # Hierarchical (temporal) processing using LSTM.
        self.lstm = nn.LSTM(
            input_size=self.flattened_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully-connected regression head
        self.fc = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, 2, freq_bins, T]
        B, C, F, T = x.size()
        cnn_out = self.cnn(x)  # e.g., [B, 64, 16, T]
        # Permute to bring time dimension to the front: [B, T, 64, 16]
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        # Flatten CNN features for each time step: [B, T, 64*16]
        cnn_out = cnn_out.reshape(B, T, -1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(cnn_out)  # lstm_out: [B, T, 2*rnn_hidden_size]
        # Use the final time step's output for regression.
        final_features = lstm_out[:, -1, :]  # [B, 2*rnn_hidden_size]
        output = self.fc(final_features)      # [B, 1]
        return output

if __name__ == "__main__":
    model = CNN_RNN_Hierarchical(freq_bins=64, rnn_hidden_size=128)
    summary(model.cuda(), (2, 64, 537))
