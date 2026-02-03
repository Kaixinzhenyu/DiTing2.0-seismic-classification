import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from utils import compute_adjacency_matrix_images,uniform,zeros,squash

class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, aggregation_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps

        self.weights = nn.Parameter(torch.Tensor(self.input_caps, input_dim, output_dim))
        
        self.aggregation_module = aggregation_module
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        # torch.Size([32, 8704, 1, 16])
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_dim)
        # torch.Size([32, 8704, 16]) 
        v,s= self.aggregation_module(u_predict)
        return v,s

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
        # will output N x OUT_CAPS x OUT_DIM  (batch,16,16,16,34)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out

class Multi_Head_Graph_Pooling(nn.Module):
    def __init__(self, num_caps_types, map_length,map_width, n_classes, output_dim, add_loop=True, improved=False, bias=True):
        super(Multi_Head_Graph_Pooling, self).__init__()
        self.n_classes = n_classes
        self.num_caps_types = num_caps_types
        self.map_length = map_length
        self.map_width = map_width
        self.output_dim = output_dim

        coord = np.zeros((map_length, map_width, 2))
        for i in range(map_length):
            for j in range(map_width):
                coord[i][j][0] = i+1
                coord[i][j][1] = j+1

        adj = torch.from_numpy(compute_adjacency_matrix_images(coord)).float()

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not improved else 2

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj_buffer = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        self.register_buffer('adj', adj_buffer)

        self.weight = nn.Parameter(torch.Tensor(output_dim, n_classes))
        self.bias = nn.Parameter(torch.Tensor(n_classes))

        uniform(self.weight)
        zeros(self.bias)
        
    def forward(self, u_predict):
        x = u_predict.view(len(u_predict)*self.num_caps_types, self.map_length*self.map_width, -1)
        s = torch.matmul(x, self.weight)
        s = torch.matmul(self.adj, s)
        s = s + self.bias
        s = torch.softmax(s, dim=1)
        #print(s.shape)#torch.Size([512, 544, 3])
        #s好像就是att 
        x = torch.matmul(s.transpose(1, 2), x)
        #print(x.shape)#torch.Size([512, 3, 16])
        u_predict = x.view(len(u_predict), -1, self.n_classes, self.output_dim)
        #print(u_predict.shape)#torch.Size([32, 16, 3, 16])
        v = u_predict.sum(dim=1)/u_predict.size()[2]
        #print(v.shape)#torch.Size([32, 3, 16])
        v = squash(v)
        return v,s
    
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

class GraCapsNet(nn.Module):
    def __init__(self, incap_dim=16, in_channel=3, out_c=256, map_length=40,map_width=72, out_dim=8640, n_classes=3, reconstructed=False):
        super(GraCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, int(out_c/incap_dim), incap_dim, kernel_size=3, stride=2)
        self.num_primaryCaps = int(out_c/incap_dim) * 18 * 34

        aggregation_module = Multi_Head_Graph_Pooling(int(out_c/incap_dim), 18,34, n_classes, 16)#(input_size-kernel_size)/stride+1
        
        self.digitCaps = CapsLayer(self.num_primaryCaps, incap_dim, n_classes, 16, aggregation_module)

        self.reconstructed = reconstructed

        self.caps = None
        self.att = None

        self.dropout = nn.Dropout()
        if reconstructed == True: self.reconstruction_net = ReconstructionNet(16, n_classes, out_dim=out_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.primaryCaps(x)
        x = self.dropout(x)
        #x(32,8704,16)
        x ,att = self.digitCaps(x)
        
        self.caps = x
        self.att = att
        probs = x.pow(2).sum(dim=2).sqrt()
        #return probs.log()
        return probs
    def reconstruct(self, target):
        return self.reconstruction_net(self.caps, target)
        '''
        target tensor([1, 1, 3, 0, 8, 3, 1, 5, 4, 7, 8, 8, 9, 0, 7, 0, 3, 2, 5, 9, 8, 8, 1, 6,
        6, 2, 3, 0, 3, 2, 7, 4, 6, 4, 0, 7, 6, 8, 8, 3, 0, 5, 2, 8, 7, 7, 3, 8,
        8, 3, 2, 2, 3, 9, 6, 5, 1, 9, 1, 4, 5, 0, 0, 6, 1, 9, 9, 2, 0, 6, 4, 1,
        0, 8, 4, 0, 9, 8, 4, 4, 3, 3, 3, 2, 9, 2, 7, 6, 5, 2, 1, 7, 1, 4, 6, 1,
        1, 1, 5, 0, 4, 3, 5, 1, 3, 0, 0, 2, 8, 3, 9, 7, 9, 1, 9, 4, 4, 3, 8, 3,
        9, 7, 2, 3, 0, 9, 1, 8]
        return torch.Size([128, 3072])正好可以变成[128, 3, 32, 32])
        '''

def gra_capsnet(num_classes=3):
    model = GraCapsNet(n_classes=num_classes,reconstructed=True)
    # reconstruction_model = ReconstructionNet(16, 10)
    # reconstruction_alpha = 0.0005
    # model = CapsNetWithReconstruction(model, reconstruction_model)
    return model




    






        
