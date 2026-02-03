import torch
from torch import nn, einsum
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, pool_kernel_size=3, pool_stride=2,
                 pool_padding=1):
        super(ConvEmbed, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            Rearrange('b d h w -> b (h w) d')
            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.conv_layers(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class CompactTransformer(nn.Module):
    def __init__(self,image_height, image_width, patch_height, patch_width, num_classes, dim=64, depth=12, heads=12, pool='cls', in_channels=3,
                 dim_head=64, dropout=0.1, emb_dropout=0.1, scale_dim=4, conv_embed=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        self.trans_layers=2
        if conv_embed:
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length(height=image_height, width=image_width)
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        # self.pool = nn.Linear(dim, 1)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        self.apply(self.init_weight)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        for layer in range(self.trans_layers):
            x = self.transformer(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        
class CapsNet(nn.Module):
    def __init__(self,routing_iterations, n_classes=3, out_dim=8640, reconstructed=False):
        super(CapsNet, self).__init__()
        #CCT Layer
        self.cctLayer = CompactTransformer(40,72,4,3, 3, conv_embed=True)
        # Primary capsule
        self.primaryCapsLayer = PrimaryCapsLayer(64, 32, 8, kernel_size=3, stride=2)  # outputs 6*6
        self.num_primaryCaps = 1024#2048#
        #self.num_primaryCaps = C × H × W / (kernel_size × stride)
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        # aggregation_module = Multi_Head_Graph_Pooling(int(out_c/incap_dim), map_size, n_classes, 16)
        # Digit capsule
        self.capsLayer = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)
        self.dropout = nn.Dropout()
        self.reconstructed = reconstructed
        if reconstructed == True: self.reconstruction_net = ReconstructionNet(16, n_classes, out_dim=out_dim)
    def forward(self, x):
        #cctlayer
        out = self.cctLayer(x)
        # batch_size, num_tokens, embed_dim = out.shape
        # H = W = int(num_tokens ** 0.5)  # 假设特征图是正方形
        # # print(out.shape)
        # # print(H)
        # #调整shape
        # out = torch.reshape(out, (batch_size, H, W, embed_dim))
        
        out = torch.reshape(out, (out.shape[0],10, 18, 64))#36,72 9,18
        out = torch.transpose(out, 1, 3)
        out = torch.transpose(out, 2, 3)
        #胶囊网络
        out = self.primaryCapsLayer(out)
        out = self.dropout(out)
        out = self.capsLayer(out)

        probs = out.pow(2).sum(dim=2).sqrt()
        return probs
    def reconstruct(self, target):
        return self.reconstruction_net(self.caps, target)

def cct_capsnet(num_classes=3):
    model = CapsNet(routing_iterations=3,n_classes=num_classes,reconstructed=True)

    return model

# if __name__ == "__main__":
#     img = torch.ones([10, 3, 36, 72])

#     model = CapsNet(routing_iterations=3,n_classes=3)
#     # parameters = filter(lambda p: p.requires_grad, cct.parameters())
#     # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     # print('Trainable Parameters in CCT: %.3fM' % parameters)

#     out = model(img)

#     print("Shape of out :", out.shape)  # [B, num_classes]
    