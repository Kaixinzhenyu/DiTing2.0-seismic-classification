import torch
import torch.nn as nn
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
class SEBasicBlock(nn.Module):
    #BasicBlock 
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * SEBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * SEBasicBlock.expansion)
            
        )
        self.se = SELayer(out_channels * SEBasicBlock.expansion, 16)
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != SEBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * SEBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * SEBasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.se(self.residual_function(x))+ self.shortcut(x))
class SEBasicBlock1D(nn.Module):
    #BasicBlock 
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * SEBasicBlock1D.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * SEBasicBlock1D.expansion)
            
        )
        self.se = SELayer1D(out_channels * SEBasicBlock1D.expansion, 16)
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != SEBasicBlock1D.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * SEBasicBlock1D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * SEBasicBlock1D.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.se(self.residual_function(x))+ self.shortcut(x))
class MultBranchSEResnet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.branch1_in_channels = 64
        self.branch2_in_channels = 64
        self.num_block= [2, 2, 2]
        # Branch 1
        self.branch1conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.branch1conv2 = self.branch1_make_layer(SEBasicBlock1D, 64, self.num_block[0], 1)
        self.branch1conv3 = self.branch1_make_layer(SEBasicBlock1D, 128, self.num_block[1], 2)
        self.branch1conv4 = self.branch1_make_layer(SEBasicBlock1D, 256, self.num_block[2], 2)
        #self.branch1conv5 = self.branch1_make_layer(SEBasicBlock1D, 512, self.num_block[3], 2)
        self.branch1avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 512)

        # Branch 2 
        self.branch2conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.branch2conv2 = self.branch2_make_layer(SEBasicBlock, 64, self.num_block[0], 1)
        self.branch2conv3 = self.branch2_make_layer(SEBasicBlock, 128, self.num_block[1], 2)
        self.branch2conv4 = self.branch2_make_layer(SEBasicBlock, 256, self.num_block[2], 2)
        #self.branch2conv5 = self.branch2_make_layer(SEBasicBlock, 512, self.num_block[3], 2)
        self.branch2avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(256, 512)

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.Linear(1, 64),nn.ReLU(inplace=True),
            nn.Linear(64, 256),nn.ReLU(inplace=True),
            nn.Linear(256, 512))
        
        # Decision layer
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),nn.ReLU(inplace=True),nn.Dropout(0.5),
            #nn.Linear(512, 512),nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128),nn.ReLU(inplace=True),
            nn.Linear(128, num_classes))
    def branch1_make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.branch1_in_channels, out_channels, stride))
            self.branch1_in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def branch2_make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.branch2_in_channels, out_channels, stride))
            self.branch2_in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    def forward(self,inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        x3 = inputs[2]

        # Branch 1
        x1 = self.branch1conv1(x1)
        x1 = self.branch1conv2(x1)
        x1 = self.branch1conv3(x1)
        x1 = self.branch1conv4(x1)
        #x1 = self.branch1conv5(x1)
        x1 = self.branch1avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        # print(x1.shape)
        x1 = self.fc1(x1)
        # Branch 2
        x2 = self.branch2conv1(x2)
        x2 = self.branch2conv2(x2)
        x2 = self.branch2conv3(x2)
        x2 = self.branch2conv4(x2)
        #x2 = self.branch2conv5(x2)
        x2 = self.branch2avgpool(x2)
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

def mult_seresnet(num_classes=3):
    return MultBranchSEResnet(num_classes=num_classes)
