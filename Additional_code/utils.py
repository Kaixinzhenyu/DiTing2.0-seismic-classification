import shutil
import torch
import os
import sys
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from torch.autograd import Variable

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
        
def compute_adjacency_matrix_images(coord, sigma=0.01):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A
def uniform(tensor, bound = 10.):
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
#特征计算
def feature_calculation(id_list,json_file):
    data=[[]]
    #循环里可以加计算过程
    for i in id_list:
        information = json_file[i]
        if 'mag' in information and information['mag']:
            data[0].append(float(information['mag']))
        # if 'se_mag' in information and information['se_mag']and information['se_mag'].replace('.', '', 1).isdigit():
        #     data[1].append(float(information['se_mag']))
        # if 'sn_mag' in information and information['sn_mag']and information['sn_mag'].replace('.', '', 1).isdigit():
        #     data[2].append(float(information['sn_mag']))
    data=torch.tensor(data)
    data = torch.transpose(data, 0, 1)
    return data
#标签编码
def lable_making(data,lable):
    if lable==0:
        tensor= torch.zeros(1, data.shape[0]) 
    elif lable==1:
        tensor = torch.ones(1, data.shape[0]) 
    elif lable==2:
        two=torch.zeros(1, data.shape[0])
        tensor = torch.full_like(two, 2)
        del two
    
    # 使用flatten()将其变为一维张量
    lable_tensor = tensor.flatten()

    return lable_tensor
def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #class_to = pred[0].cpu().numpy()
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res #,class_to

def save_checkpoint(state,is_best,dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,checkpoint_path,Num_classes,order):
    filename = os.path.join(checkpoint_path,'{}_{}_{}_{}_{}_{}_{}.pth'.format(dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,Num_classes,order))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_best_{}_{}_{}_{}_{}_{}_{}.pth'.format(dataset_choose,model_choose,Batch_size,learning_rate,Weight_decay,Num_classes,order)))

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_network(dataset_choose,net,Num_classes=3,Reconstruct=False):
    if net == 'cnn1d':
        from models.cnn1d import cnn1d
        model = cnn1d(num_classes=Num_classes)
    elif net == 'cnn2d':
        from models.cnn2d import cnn2d
        model = cnn2d(dataset_choose,num_classes=Num_classes)
    elif net == 'vgg16':
        from models.vgg import vgg16_bn
        model = vgg16_bn(dataset_choose,num_classes=Num_classes)
    elif net == 'vgg13':
        from models.vgg import vgg13_bn
        model = vgg13_bn(dataset_choose,num_classes=Num_classes)
    elif net == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn(dataset_choose,num_classes=Num_classes)
    elif net == 'vgg19':
        from models.vgg import vgg19_bn
        model = vgg19_bn(dataset_choose,num_classes=Num_classes)
    elif net == 'vgg1d':
        from models.vgg1d import vgg19
        model = vgg19(dataset_choose,num_classes=Num_classes)
    elif net == 'googlenet':
        from models.googlenet import googlenet
        model = googlenet(num_classes=Num_classes)
    elif net == 'alexnet':
        from models.alexnet import alexnet
        model = alexnet(dataset_choose,num_classes=Num_classes)
    elif net == 'googlenet1d':
        from models.googlenet1d import googlenet1d
        model = googlenet1d(num_classes=Num_classes)
    elif net == 'alexnet1d':
        from models.alexnet1d import alexnet
        model = alexnet(num_classes=Num_classes)
    # elif net == 'inceptionv3':
    #     from models.inceptionv3 import inceptionv3
    #     model = inceptionv3()
    # elif net == 'inceptionv4':
    #     from models.inceptionv4 import inceptionv4
    #     model = inceptionv4()

    elif net == 'resnet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=Num_classes)
    elif net == 'resnet1d':
        from models.resnet1d import resnet50
        model = resnet50(num_classes=Num_classes)
    elif net == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=Num_classes)
    
    elif net == 'resnext50':
        from models.resnext import resnext50
        model = resnext50(num_classes=Num_classes)
    elif net == 'cnn_capsnet_reconstruct':
        from models.cnn_capsnet_reconstruct import cnn_capsnet
        model = cnn_capsnet(model_choose="cnn_capsnet", num_classes=3, reconstructed=True)
    elif net == 'res_capsnet_reconstruct':
        from models.cnn_capsnet_reconstruct import cnn_capsnet
        model = cnn_capsnet(model_choose="res_capsnet", num_classes=3, reconstructed=True)
    elif net == 'cnn_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'res_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'cbam_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'cbamres_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'res18_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'seres_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net == 'cct_capsnet':
        from models.cct_capsnet import cct_capsnet
        model = cct_capsnet(num_classes=Num_classes)
    elif net == 'gra_capsnet':
        from models.gra_capsnet import gra_capsnet
        model = gra_capsnet(num_classes=Num_classes)
    elif net == 'se_capsnet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose=net)
    elif net=="vit":
        from models.vit import ViT
        # ViT for cifar10
        model = ViT(image_height=40, image_width=72, patch_height=10, patch_width=12,
        num_classes = 3,dim =256,depth = 6,heads = 16,mlp_dim = 256,dropout = 0.1,
        emb_dropout = 0.1)
    # elif net=="vit":
    #     from models.vit import ViT
    #     # ViT for cifar10
    #     model = ViT(image_height=72, image_width=72, patch_height=6, patch_width=6,
    #     num_classes = 3,dim =256,depth = 6,heads = 16,mlp_dim = 256,dropout = 0.1,
    #     emb_dropout = 0.1)
    elif net=="swin":
        from models.swin import swin_t
        model = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    elif net=="cct":
        from models.cct import CompactTransformer
        model = CompactTransformer(image_height=40, image_width=72, 
                                    patch_height=10, patch_width=12, 
                                    num_classes=Num_classes, conv_embed=True)
    # elif net=="cct":
    #     from models.cct import CompactTransformer
    #     model = CompactTransformer(image_height=72, image_width=72, 
    #                                 patch_height=6, patch_width=6, 
    #                                 num_classes=Num_classes, conv_embed=True)                             
    elif net=="mult_vgg":
        from models.mult_branch_vgg import mult_vgg
        model = mult_vgg(num_layers=11, num_classes=Num_classes)
    elif net=="mult_resnet":
        from models.mult_branch_resnet import mult_resnet
        model = mult_resnet(num_classes=Num_classes)
    elif net=="mult_seresnet":
        from models.mult_branch_seresnet import mult_seresnet
        model = mult_seresnet(num_classes=Num_classes)
    elif net=="mult_cbamresnet":
        from models.mult_branch_cbamresnet import mult_cbamresnet
        model = mult_cbamresnet(num_classes=Num_classes)
    elif net=="mult_attention":
        from models.mult_branch_attention import mult_attention
        model = mult_attention(num_classes=Num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return model