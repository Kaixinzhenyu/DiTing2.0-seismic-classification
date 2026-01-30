import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# you need to download the models to ~/.torch/models
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }

class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv1d(3, 96, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1), # shape is 27 x 27 x 64

            nn.Conv1d(96, 256, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1), # shape is 13 x 13 x 192

            nn.Conv1d(256, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(),

            nn.Conv1d(384, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),

            nn.Conv1d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1) # shape is 6 x 6 x 256
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(190976, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x


def alexnet(num_classes=2):
    """
    AlexNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(num_classes=num_classes)

    return model