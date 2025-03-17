import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """Remove padding in 2D for temporal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock1D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock1D, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet1D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet1D, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock1D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MustafaNet1DTCN(nn.Module):
    def __init__(self, num_classes=15, anticipated_frames=8):
        super(MustafaNet1DTCN, self).__init__()
        self.anticipated_frames = anticipated_frames
        # TemporalConvNet2D: takes 2048-channel features over temporal window as input
        self.tcn_local = TemporalConvNet1D(num_inputs=2048, num_channels=[256, 512, 512, 256], kernel_size=3, dropout=0.2)
        
        # Final regression layer
        self.regression = nn.Conv1d(in_channels=256, out_channels=num_classes * anticipated_frames, kernel_size=1)

    def forward(self, x):
        # Expected input shape: [batch, window, 2048]
        x = x.permute(0, 2, 1)
        x = self.tcn_local(x)  # Apply 2D TCN
        x = self.regression(x)  # Apply final regression layer to get class scores
        x = x.view(x.size(0), self.anticipated_frames, -1, x.size(2))
        return x.mean(dim=3) # Output shape: [batch, num_classes]
