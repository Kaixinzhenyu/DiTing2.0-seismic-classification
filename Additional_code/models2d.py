import torch.nn as nn
from torchvision import models
import torch
#HeartNet
class CNN2d(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN2d, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(64, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(128, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm2d(256, eps=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 创建一个虚拟输入以获取期望的特征图大小
        dummy_input = torch.randn(1, 3, 257, 24)
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


class MobileNetV2(models.MobileNetV2):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)


class AlexNet(models.AlexNet):
    def __init__(self, num_classes=8):
        super().__init__(num_classes=num_classes)


def VGG16(num_classes=8):
    model = models.vgg16_bn()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def ResNet18(num_classes=8):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet34(num_classes=8):
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ShuffleNet(num_classes=8):
    model = models.shufflenet_v2_x1_0()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# def EfficientNetB4(num_classes=8):
#     model = efficientnet.from_name("efficientnet-b4")
#     model._fc = nn.Linear(model._fc.in_features, num_classes)
#     return model
