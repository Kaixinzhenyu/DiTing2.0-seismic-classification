import torch
from torch import nn

class CNN2d(nn.Module):
    def __init__(self, dataset_choose,num_classes=3):
        super(CNN2d, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=0.001),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 创建一个虚拟输入以获取期望的特征图大小
        if dataset_choose=='STFT':
            dummy_input = torch.randn(1, 3, 26, 118)
        elif dataset_choose=='STFT_72_72':
            dummy_input = torch.randn(1, 3, 72, 72)
        elif dataset_choose=='GADF':
            dummy_input = torch.randn(1, 3, 128, 128)
        elif dataset_choose=='MFCC_36_72':
            dummy_input = torch.randn(1, 3, 36, 72)
        elif dataset_choose=='MFCC':
            dummy_input = torch.randn(1, 3, 39, 24)
        dummy_output = self.features(dummy_input)
        dummy_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(dummy_output_size, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 256),  # 新增一层全连接层
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def cnn2d(dataset_choose,num_classes=3):
    return CNN2d(dataset_choose,num_classes=num_classes)
