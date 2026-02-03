import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# you need to download the models to ~/.torch/models
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }

class AlexNet(nn.Module):
    def __init__(self, dataset_choose,num_classes=3):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1), # shape is 27 x 27 x 64

            nn.Conv2d(96, 256, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1), # shape is 13 x 13 x 192

            nn.Conv2d(256, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1) # shape is 6 x 6 x 256
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
        elif dataset_choose=='MFCC_40_72':
            dummy_input = torch.randn(1, 3, 40, 72)
        elif dataset_choose=='MFCC':
            dummy_input = torch.randn(1, 3, 39, 24)
        elif dataset_choose=='mfcc':
            dummy_input = torch.randn(1, 3, 40, 72)
        elif dataset_choose=='stft':
            dummy_input = torch.randn(1, 3, 72, 72)
        dummy_output = self.features(dummy_input)
        dummy_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(dummy_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(dataset_choose,num_classes=3):
    """
    AlexNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(dataset_choose,num_classes=num_classes)

    return model