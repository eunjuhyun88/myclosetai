"""
Common neural network blocks for geometric matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommonBottleneckBlock(nn.Module):
    """공통 Bottleneck 블록"""
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_resnet_layer(block_class, inplanes, planes, blocks, stride=1, dilation=1, downsample=None):
    """ResNet 레이어 생성"""
    layers = []
    layers.append(block_class(inplanes, planes, stride, dilation, downsample))
    inplanes = planes * 4
    for _ in range(1, blocks):
        layers.append(block_class(inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers)


class CommonConvBlock(nn.Module):
    """공통 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CommonInitialConv(nn.Module):
    """공통 초기 컨볼루션"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CommonFeatureExtractor(nn.Module):
    """공통 특징 추출기"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = CommonConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        return self.conv(x)


class CommonAttentionBlock(nn.Module):
    """공통 어텐션 블록"""
    
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CommonGRUConvBlock(nn.Module):
    """공통 GRU 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gru = nn.GRU(out_channels, out_channels, batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.size()
        conv_out = self.conv(x)
        # Reshape for GRU: (B, C, H*W) -> (B, H*W, C)
        gru_input = conv_out.view(b, c, -1).permute(0, 2, 1)
        gru_out, _ = self.gru(gru_input)
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        return gru_out.permute(0, 2, 1).view(b, c, h, w)
