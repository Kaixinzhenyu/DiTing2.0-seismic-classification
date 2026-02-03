"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""
import torch
import torch.nn as nn

class Inception1D(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv1d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm1d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv1d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm1d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv1d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 5x5 conv branch
        # 使用两个3x3卷积层堆叠以获得相同的感受野
        self.b3 = nn.Sequential(
            nn.Conv1d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm1d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv1d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm1d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv1d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm1d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pooling -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class GoogleNet1D(nn.Module):

    def __init__(self, num_class=3):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )

        # Inception 模块
        self.a3 = Inception1D(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception1D(256, 128, 128, 192, 32, 96, 64)

        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception1D(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception1D(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception1D(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception1D(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception1D(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception1D(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception1D(832, 384, 192, 384, 48, 128, 128)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        # 全局平均池化和 dropout
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def googlenet1d(num_classes=3):
    return GoogleNet1D(num_class=num_classes)
