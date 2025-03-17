import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp2d(nn.Module):
    """Padding 제거를 위한 모듈."""
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()

class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MustafaNet2DTCN(nn.Module):
    """Model using 3D Convolution to handle sequences of images"""
    def __init__(self):
        super(MustafaNet2DTCN, self).__init__()
        # 3D convolution: input channels=3 (RGB), output channels=16, kernel=(4, 3, 3), padding=(0, 1, 1)
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(4, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU()
        
        # 2D TCN layers after the 3D convolution (output channels=16 from the conv3d)
        self.tcn_local = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Final layer to output class predictions (27 classes in your case)
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1)

    def forward(self, x):
        # Input shape: [batch_size, window_size, channels, height, width]
        # Permute to [batch_size, channels, window_size, height, width] for Conv3D
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, window_size, height, width)

        x = self.conv3d(x)  # Apply 3D convolution
        x = self.relu(x)

        # Squeeze the temporal dimension (now it's (batch_size, channels, height, width))
        x = x.squeeze(2)

        # Apply the 2D TCN layers
        x = self.tcn_local(x)

        # Output layer
        x = self.conv_out(x)

        # Perform global average pooling over height and width (reduce to [batch_size, num_classes])
        x = x.mean(dim=(2, 3))
        return x