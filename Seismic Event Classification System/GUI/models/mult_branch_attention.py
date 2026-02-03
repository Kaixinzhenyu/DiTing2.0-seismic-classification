import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    #BasicBlock 
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
class BasicBlock1D(nn.Module):
    #BasicBlock 
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class MultBranchlAttention(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.branch1_in_channels = 64
        self.branch2_in_channels = 64
        self.num_block= [2, 2, 2, 2]
        # Branch 1
        self.branch1conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.branch1conv2 = self.branch1_make_layer(BasicBlock1D, 64, self.num_block[0], 1)
        self.branch1conv3 = self.branch1_make_layer(BasicBlock1D, 128, self.num_block[1], 2)
        self.branch1conv4 = self.branch1_make_layer(BasicBlock1D, 256, self.num_block[2], 2)
        self.branch1conv5 = self.branch1_make_layer(BasicBlock1D, 512, self.num_block[3], 2)
        self.branch1avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 512)

        # Branch 2 
        self.branch2conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.branch2conv2 = self.branch2_make_layer(BasicBlock, 64, self.num_block[0], 1)
        self.branch2conv3 = self.branch2_make_layer(BasicBlock, 128, self.num_block[1], 2)
        self.branch2conv4 = self.branch2_make_layer(BasicBlock, 256, self.num_block[2], 2)
        self.branch2conv5 = self.branch2_make_layer(BasicBlock, 512, self.num_block[3], 2)
        self.branch2avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(512, 512)

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.Linear(1, 64),nn.ReLU(),
            nn.Linear(64, 512))
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # Decision layer
        self.fc_fcfinal = nn.Sequential(
            nn.Linear(512, 128),nn.ReLU(),
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
        x1 = self.branch1conv5(x1)
        x1 = self.branch1avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)

        # Branch 2
        x2 = self.branch2conv1(x2)
        x2 = self.branch2conv2(x2)
        x2 = self.branch2conv3(x2)
        x2 = self.branch2conv4(x2)
        x2 = self.branch2conv5(x2)
        x2 = self.branch2avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)
        # Branch 3
        # 添加一个维度，将其扩展为 [batch_size, 1, 1]
        x3 = self.branch3(x3)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x1.unsqueeze(0).shape)
        # print(x2.unsqueeze(0).shape)
        # print(x3.unsqueeze(0).shape)
        # Concatenate
        x = torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)), dim=0)
        # Classifier layer
        x = self.transformerEncoder(x)
        x = torch.mean(x, dim=0)
        x = self.fc_fcfinal(x)
        return x

def mult_attention(num_classes=3):
    return MultBranchlAttention(num_classes=num_classes)
