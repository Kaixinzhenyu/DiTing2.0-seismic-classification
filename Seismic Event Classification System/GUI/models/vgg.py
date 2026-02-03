"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, dataset_choose,num_class=3):
        super().__init__()
        self.features = features
        # 创建一个虚拟输入以获取期望的特征图大小
        if dataset_choose=='STFT':
            dummy_input = torch.randn(1, 3, 26, 118)
        elif dataset_choose=='GADF':
            dummy_input = torch.randn(1, 3, 128, 128)
        elif dataset_choose=='MFCC':
            dummy_input = torch.randn(1, 3, 39, 24)
        elif dataset_choose=='MFCC_36_72':
            dummy_input = torch.randn(1, 3, 36, 72)
        elif dataset_choose=='STFT_72_72':
            dummy_input = torch.randn(1, 3, 72, 72)
        elif dataset_choose=='MFCC_40_72':
            dummy_input = torch.randn(1, 3, 40, 72)
        elif dataset_choose=='mfcc':
            dummy_input = torch.randn(1, 3, 40, 72)
        elif dataset_choose=='stft':
            dummy_input = torch.randn(1, 3, 72, 72)
        dummy_output = self.features(dummy_input)
        dummy_output_size = dummy_output.view(dummy_output.size()[0], -1)
        out_features = dummy_output_size.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512, 512),nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),nn.ReLU(),
            nn.Linear(128, num_class))

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(dataset_choose,num_classes=3):
    return VGG(make_layers(cfg['A'], batch_norm=True),dataset_choose=dataset_choose,num_class=num_classes)

def vgg13_bn(dataset_choose,num_classes=3):
    return VGG(make_layers(cfg['B'], batch_norm=True),dataset_choose=dataset_choose,num_class=num_classes)

def vgg16_bn(dataset_choose,num_classes=3):
    return VGG(make_layers(cfg['D'], batch_norm=True),dataset_choose=dataset_choose,num_class=num_classes)

def vgg19_bn(dataset_choose,num_classes=3):
    return VGG(make_layers(cfg['E'], batch_norm=True),dataset_choose=dataset_choose,num_class=num_classes)


