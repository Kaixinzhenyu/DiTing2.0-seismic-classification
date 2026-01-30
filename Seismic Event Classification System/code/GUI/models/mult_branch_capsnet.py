import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
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
def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x

class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)
        return v
   
class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v

class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out

class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""
    def __init__(self,routing_iterations,model_choose,n_classes=3):
        super(CapsNet, self).__init__()
        self.model_choose=model_choose
        self.branch2_in_channels = 64
        # Conv2d layer/cct
        self.conv2d = nn.Conv2d(3, 256, kernel_size=3, stride=1)
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = self.make_layer(BasicBlock, 64, 2, 1)
        self.conv3 = self.make_layer(BasicBlock, 128, 2, 2)
        self.conv4 = self.make_layer(BasicBlock, 256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = nn.Linear(256, 256)
        self.shortcut = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256)
            )
        self.se = SELayer(256, 16)
        self.relu = nn.ReLU(inplace=True)
        # Primary capsule
        self.primaryCapsLayer = PrimaryCapsLayer(256, 16, 16, kernel_size=3, stride=2) 
        if self.model_choose == 'res_capsnet':
            self.num_primaryCaps = 9520
        elif self.model_choose == 'cnn_capsnet':
            self.num_primaryCaps = 8704
        elif self.model_choose == 'seres_capsnet':
            self.num_primaryCaps = 9520
        elif self.model_choose == 'res18_capsnet':
            self.num_primaryCaps = 512
        # self.num_primaryCaps = int(out_c/incap_dim) * map_size * map_size
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        # Digit capsule
        self.capsLayer = CapsLayer(self.num_primaryCaps, 16, n_classes, 16, routing_module)
        self.dropout = nn.Dropout()
        if reconstructed == True: self.reconstruction_net = ReconstructionNet(16, n_classes, 7776)
    def make_layer(self, block, out_channels, num_blocks, stride):
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
    def forward(self, x):
        if self.model_choose == 'res_capsnet':
            out = self.relu(self.residual_function(x) + self.shortcut(x))
            out = self.dropout(out)
        elif self.model_choose == 'cnn_capsnet':
            out = self.relu(self.conv2d(x))
            out = self.dropout(out)
        elif self.model_choose == 'seres_capsnet':
            out = self.relu(self.se(self.residual_function(x))+ self.shortcut(x))
            out = self.dropout(out)
        elif self.model_choose == 'res18_capsnet':
            out = self.conv4(self.conv3(self.conv2(self.conv1(x))))

        out = self.primaryCapsLayer(out)
        # print(out.shape)
        out = self.dropout(out)
        out = self.capsLayer(out)
        # print(out.shape)
        probs = out.pow(2).sum(dim=2).sqrt()
        return out,probs
    def reconstruct(self, target):
        return self.reconstruction_net(self.caps, target)
class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10, out_dim=784):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, out_dim)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda: mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


def cnn_capsnet(model_choose,num_classes=3):
    model = CapsNet(model_choose=model_choose,routing_iterations=3,n_classes=num_classes)
    reconstruction_model = ReconstructionNet(16, 3)
    
    mult_model = CapsNetWithReconstruction(model, reconstruction_model)
    
    return mult_model


if __name__ == "__main__":
    img = torch.ones([1, 3, 36, 72])
    models=cnn_capsnet('cnn_capsnet',3)

    out = models(img)

    print("Shape of out :", out.shape)  # [B, num_classes]