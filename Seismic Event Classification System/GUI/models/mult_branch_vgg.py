import torch.nn as nn
import math
import torch
import torch.nn.functional as F

def make_layers_vgg(branch,num_layers,batch_norm=False):
    cfg = {
    '11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    '13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    '19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    layers = []
    if branch==1:
        input_channel = 3
        for l in cfg[str(num_layers)]:
            if l == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                continue
            layers += [nn.Conv1d(input_channel, l, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm1d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l
    elif branch==2:
        input_channel = 3
        for l in cfg[str(num_layers)]:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l

    return nn.Sequential(*layers)

class MultBranchVgg(nn.Module):
    def __init__(self,num_layers,num_classes):
        super().__init__()
        # Branch 1
        self.branch1 = make_layers_vgg(branch=1,num_layers=num_layers,batch_norm=True)
        self.fc1 = nn.Linear(47616, 512)

        # Branch 2 
        self.branch2 = make_layers_vgg(branch=2,num_layers=num_layers,batch_norm=True)
        self.fc2 = nn.Linear(1024, 512)

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.Linear(1, 64),nn.ReLU(),
            nn.Linear(64, 512))
        
        # Decision layer
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512, 512),nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),nn.ReLU(),
            nn.Linear(128, num_classes))
    def forward(self,inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        x3 = inputs[2]

        # Branch 1
        x1 = self.branch1(x1)
        x1 = x1.view(x1.size(0), -1)
        # print(x1.shape)
        x1 = self.fc1(x1)

        # Branch 2
        x2 = self.branch2(x2)
        x2 = x2.view(x2.size(0), -1)
        # print(x2.shape)
        x2 = self.fc2(x2)
        # Branch 3
        # 添加一个维度，将其扩展为 [batch_size, 1, 1]
        x3 = self.branch3(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # Concatenate
        x = torch.cat((x1,x2,x3), dim=1)
        # print(x.shape)
        # Classifier layer
        x = self.classifier(x)
        return x

def mult_vgg(num_layers=11,num_classes=3):
    '''
    num_layers=11 13 16 19
    '''
    return MultBranchVgg(num_layers=num_layers,num_classes=num_classes)
