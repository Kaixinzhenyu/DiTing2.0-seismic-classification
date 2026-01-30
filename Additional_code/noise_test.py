import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import json
import os
import tqdm
import time
from data_loader import test_dataloader,noise_test_dataloader,two_class_test_dataloader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate,get_network

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

#数据根路径(3,expand)
dataset_id='expand'
#数据集选择 ('MFCC_40_72','STFT_72_72','mfcc',stft)
dataset_choose='mfcc'
noise='True'
#模型选择 (cnn1d,resnet1d,vgg1d,alexnet1d,cnn2d,googlenet,googlenet1d,alexnet,resnet18,resnet50,vit,cct,vit_small,swin,mult_cnn,vgg11,
    #         cnn_capsnet,res_capsnet,seres_capsnet,res18_capsnet,
    #         cct_capsnet,gra_capsnet)
model_choose='cct_capsnet'
checkpoint_basic_path='/root/autodl-tmp/Python projects/my_project/checkpoint'
checkpoint_name='model_best_mfcc_cct_capsnet_32_0.0001_0.0001_3_44'
# 模型权重保存的路径  
checkpoint_path = checkpoint_basic_path+'/{}/'.format(dataset_id)+checkpoint_name+'.pth'

#数据路径  
base_path='/root/autodl-tmp/Python projects/Dataset/'+'{}'.format(dataset_id)

# Load the checkpoint from file
checkpoint = torch.load(checkpoint_path)
#类别数
Num_classes=3
classes = ['natural','non_natural','noise']

#定义模型
model = get_network(dataset_choose=dataset_choose,net=model_choose,Num_classes=Num_classes)
model.cuda()
# Load the model's state dictionary
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#数据集加载     

dataset_id =='expand' and dataset_choose== 'mfcc' and noise=='True'
test_loader =noise_test_dataloader(natural_test_path=base_path+'/mfcc/natural_test_MFCC.pt',
                                ep_test_path=base_path+'/mfcc/ep_test_MFCC.pt',
                                ss_test_path=base_path+'/mfcc/ss_test_MFCC.pt',
                                noise_test_path=base_path+'/mfcc/noise_test_MFCC.pt'
                                )
natural_id_list = torch.load(base_path+'/natural_test_id.pt')
ep_id = torch.load(base_path+'/ep_test_id.pt')
ss_id = torch.load(base_path+'/ss_test_id.pt')   
non_natural_id_list = ep_id+ss_id 
 
Total_Pred=[]
Total_Target=[]

# 选择指定的批次（索引从0开始）
for i, (input, target) in enumerate(test_loader):
    input = input.cuda().to(torch.float32)#torch.Size([1, 3, 3000])
    target = target.cuda().to(torch.long)

    # compute output
    output= model(input)
    value,pred=torch.max(output,1)
    # Check if the prediction is correct
    Total_Pred.append(pred.tolist())
    Total_Target.append(target.tolist())
# 使用numpy.squeeze去除维度为1的维度
Total_Pred = np.squeeze(Total_Pred).tolist()
Total_Target = np.squeeze(Total_Target).tolist()

# 计算准确率
accuracy = accuracy_score(Total_Target, Total_Pred)
print('准确率: {:.2%}'.format(accuracy))

# 计算精确率
precision = precision_score(Total_Target, Total_Pred, average='weighted')  # 'weighted' for multiclass problems
print('精确率: {:.2%}'.format(precision))

# 计算召回率
recall = recall_score(Total_Target, Total_Pred, average='weighted')  # 'weighted' for multiclass problems
print('召回率: {:.2%}'.format(recall))

# 计算 F1-score
f1 = f1_score(Total_Target, Total_Pred, average='weighted')  # 'weighted' for multiclass problems
print('F1-score: {:.2%}'.format(f1))

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    # 添加颜色条
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=20)  # 设置颜色条的字体大小

    if target_names is not None:
        #plt.colorbar()
        #plt.imshow(data, cmap=cmap, vmin=0, vmax=1) 
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=20)
        plt.yticks(tick_marks, target_names, fontsize=20)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # [mwen]

    thresh = cm.max() / 2 # [mwen]
    thresh_norm = cm_norm.max() / 1.5 # [mwen]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm_norm[i, j]) + "\n{:,}".format(cm[i, j]), # [mwen]
                     horizontalalignment="center", verticalalignment="center", fontsize=20,
                     color="white" if cm_norm[i, j] > thresh_norm else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",fontsize=20,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20, color='darkblue') # fontweight='bold'
    plt.xlabel('Predicted label\nAccuracy: {:.2%}; Precision: {:.2%}; Recall: {:.2%}; F1-score: {:.2%}'.format(accuracy, precision, recall, f1), fontsize=16, color='darkblue')
    # 保存图像为矢量图，指定保存路径
    plt.savefig('/root/autodl-tmp/test_result/noise/'+'{}'.format(checkpoint_name)+'.svg', format='svg', dpi=300)
    plt.show()

#画混淆矩阵
cm = confusion_matrix(Total_Target, Total_Pred)
plot_confusion_matrix(cm, target_names=classes, title='confusion_matrix')



