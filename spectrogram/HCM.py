import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# Custom LSTM module that returns only the output tensor and hides its inner modules from children()
class LSTMOutput(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
    def forward(self, x):
        # Forward pass: return only the output tensor (ignore hidden states)
        output, _ = self.lstm(x)
        return output
    def children(self):
        # Override children() so that torchsummary doesn't try to descend into the LSTM's internals.
        return iter([])
    def _apply(self, fn):
        # Ensure that when device-changing functions are applied, they affect the inner LSTM as well.
        self.lstm._apply(fn)
        return super()._apply(fn)

class HCM(nn.Module):
    def __init__(self, freq_bins=64, rnn_hidden_size=128):
        """
        freq_bins: Number of frequency channels in the cochleogram (e.g., 64)
        rnn_hidden_size: Hidden size for the LSTM
        """
        super().__init__()
        
        # Local feature extraction using CNN.
        # Input shape: [B, 2, freq_bins, T]
        self.local_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),  # preserves spatial size
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # reduces frequency dimension by 2; output: [B, 32, freq_bins//2, T]
        )
        
        # Calculate flattened feature size per time step after the CNN.
        self.flattened_feature_size = 32 * (freq_bins // 2)
        
        # Create the custom LSTM module.
        self.lstm = LSTMOutput(
            input_size=self.flattened_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Final fully-connected head for regression.
        # LSTM is bidirectional so the final feature dimension is 2*rnn_hidden_size.
        self.fc = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Maps output to [0, 1]
        )
    
    def forward(self, x):
        """
        x: Input tensor of shape [B, 2, freq_bins, T]
        """
        B, C, F, T = x.size()  # C=2, F=freq_bins
        # Apply the local CNN: output shape: [B, 32, F//2, T]
        cnn_out = self.local_cnn(x)
        # Permute to bring the time dimension to the front: [B, T, 32, F//2]
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        # Flatten the channel and frequency dimensions: new shape: [B, T, 32*(F//2)]
        cnn_out = cnn_out.reshape(B, T, -1)
        
        # Process the sequence with the custom LSTM.
        # lstm_out: [B, T, 2*rnn_hidden_size]
        lstm_out = self.lstm(cnn_out)
        # For simplicity, take the output at the final time step.
        final_features = lstm_out[:, -1, :]  # shape: [B, 2*rnn_hidden_size]
        
        # Final prediction through the fully-connected head.
        output = self.fc(final_features)  # shape: [B, 1]
        return output

if __name__ == "__main__":
    # Instantiate the model.
    model = HCM(freq_bins=64, rnn_hidden_size=128)
    # Move the model to GPU.
    model = model.cuda()
    # Print a summary of the model.
    # Expected input shape: (2, 64, 537) -> 2 channels, 64 frequency bins, 537 time steps.
    summary(model, (2, 64, 537))
