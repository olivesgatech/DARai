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
    """2D TCN에서의 기본 블록 (Temporal Block)"""
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
    """2D TCN 네트워크 정의"""
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
    """2D TCN을 사용하는 MustafaNet"""
    def __init__(self):
        super(MustafaNet2DTCN, self).__init__()
        self.tcn_local = TemporalConvNet2D(num_inputs=4, num_channels=[16, 32, 64, 128], kernel_size=3, dropout=0.2)
        self.conv_out = nn.Conv2d(in_channels=128, out_channels=6, kernel_size=1)  # 클래스 개수만큼 출력 채널 설정

    def forward(self, x):
        out = self.tcn_local(x)
        out = self.conv_out(out)
        return out
