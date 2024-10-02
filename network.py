'''
-*- coding:utf-8 -*-
@File:network.py
'''
import torch
from torch import nn

# define convolutional block
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0, bias=False):
        super(ConvolutionalBlock, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sub_module(x)
        return x

# define residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionalBlock(out_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.sub_module(x) + x
        return x

# define 1*4096*8 down sampling block
class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalBlock(in_channels, out_channels, kernel_size=3, stride=(1, 2), padding=1),
        )

    def forward(self, x):
        x = self.sub_module(x)
        return x

# define model
class SpinEchoNet(nn.Module):
    def __init__(self):
        super(SpinEchoNet, self).__init__()

        # define input module--down_sampling module in indirect dimensions
        self.trunk_4096 = nn.Sequential(
            ConvolutionalBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            DownSamplingBlock(in_channels=16, out_channels=32),

            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            DownSamplingBlock(in_channels=32, out_channels=64),

            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),

            DownSamplingBlock(in_channels=64, out_channels=128),

            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
        )

        # define output--dimension_reducing module in channel dimensions
        self.out = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=8, out_channels=8),
            ResidualBlock(in_channels=8, out_channels=8),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=4, out_channels=4),
            ResidualBlock(in_channels=4, out_channels=4),
            nn.Softmax(dim=2),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0),
        )

        self.div = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  
        )


    def forward(self, x):
        per_out = []
        x = self.trunk_4096(x)
        per_out.append(x)
        x = self.out(x)
        per_out.append(x)
        return x, per_out