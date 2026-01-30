import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from torch.utils.data import random_split
import json
from utils import feature_calculation,lable_making
class Mult_branch_loader(torch.utils.data.Dataset):
    def __init__(self, dataset_id, dataset_choose,base_path):
        super(Mult_branch_loader, self).__init__()
        self.waveform_natural_train = torch.load(base_path+'/tensor dataset/natural_train_dataset.pt')
        self.waveform_ep_train = torch.load(base_path+'/tensor dataset/ep_train_dataset.pt')
        self.waveform_ss_train = torch.load(base_path+'/tensor dataset/ss_train_dataset.pt')
        if dataset_choose == 'STFT_72_72':
            self.image_natural_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/natural_train_STFT.pt')
            self.image_ep_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ep_train_STFT.pt')
            self.image_ss_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ss_train_STFT.pt')
        elif dataset_choose == 'MFCC_36_72':
            self.image_natural_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/natural_train_MFCC.pt')
            self.image_ep_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ep_train_MFCC.pt')
            self.image_ss_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ss_train_MFCC.pt')
        #feature_calculation计算地震特征 添加第三分支
        id_natural_train = torch.load(base_path+'/tensor dataset/natural_train_id.pt')
        natural_json_file = json.load(open('/home/zzy/Python projects/Dataset/DiTing2.0/CENC_DiTingv2_natural_earthquake.json', 'r'))
        self.data_natural_train=feature_calculation(id_natural_train,natural_json_file)
        del id_natural_train,natural_json_file
        id_ep_train = torch.load(base_path+'/tensor dataset/ep_train_id.pt')
        non_natural_json_file = json.load(open('/home/zzy/Python projects/Dataset/DiTing2.0/CENC_DiTingv2_non_natural_earthquake.json', 'r'))
        self.data_ep_train=feature_calculation(id_ep_train,non_natural_json_file)
        del id_ep_train
        id_ss_train = torch.load(base_path+'/tensor dataset/ss_train_id.pt')
        self.data_ss_train=feature_calculation(id_ss_train,non_natural_json_file)
        del id_ss_train,non_natural_json_file

        self.waveform_branch = []
        self.image_branch = []
        self.data_branch = []
        self.labels = []
        for i in range(self.waveform_natural_train.shape[0]):
            self.waveform_branch.append(self.waveform_natural_train[i])
            self.image_branch.append(self.image_natural_train[i])
            self.data_branch.append(self.data_natural_train[i])
            self.labels.append(0)
        for j in range(self.waveform_ep_train.shape[0]):
            self.waveform_branch.append(self.waveform_ep_train[j])
            self.image_branch.append(self.image_ep_train[j])
            self.data_branch.append(self.data_ep_train[j])
            self.labels.append(1)
        for k in range(self.waveform_ss_train.shape[0]):
            self.waveform_branch.append(self.waveform_ss_train[k])
            self.image_branch.append(self.image_ss_train[k])
            self.data_branch.append(self.data_ss_train[k])
            self.labels.append(2)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        w = self.waveform_branch[idx].float()
        i = self.image_branch[idx].float()
        d = self.data_branch[idx].float()
        l = self.labels[idx]
        return (w, i, d), l
#数据根路径(1,2,3)
dataset_id=1
#数据集选择 ('1d','GADF','STFT_26_118','STFT_72_72','MFCC_39_24','MFCC_36_72')
dataset_choose='STFT_72_72'
#数据路径  
base_path='/home/zzy/Python projects/Dataset/'+'{}'.format(dataset_id)
train_dataset=Mult_branch_loader(dataset_id, dataset_choose, base_path)
#划分训练集与验证集
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

# waveform_natural_train = torch.load(base_path+'/tensor dataset/natural_train_dataset.pt')
# waveform_ep_train = torch.load(base_path+'/tensor dataset/ep_train_dataset.pt')
# waveform_ss_train = torch.load(base_path+'/tensor dataset/ss_train_dataset.pt')
# if dataset_choose == 'STFT_72_72':
#     image_natural_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/natural_train_STFT.pt')
#     image_ep_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ep_train_STFT.pt')
#     image_ss_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ss_train_STFT.pt')
# elif dataset_choose == 'MFCC_36_72':
#     image_natural_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/natural_train_MFCC.pt')
#     image_ep_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ep_train_MFCC.pt')
#     image_ss_train = torch.load(base_path+'/processed_data/'+dataset_choose+'/ss_train_MFCC.pt')
# #feature_calculation计算地震特征 添加第三分支
# id_natural_train = torch.load(base_path+'/tensor dataset/natural_train_id.pt')
# natural_json_file = json.load(open('/home/zzy/Python projects/Dataset/DiTing2.0/CENC_DiTingv2_natural_earthquake.json', 'r'))
# data_natural_train=feature_calculation(id_natural_train,natural_json_file)
# del id_natural_train,natural_json_file
# id_ep_train = torch.load(base_path+'/tensor dataset/ep_train_id.pt')
# non_natural_json_file = json.load(open('/home/zzy/Python projects/Dataset/DiTing2.0/CENC_DiTingv2_non_natural_earthquake.json', 'r'))
# data_ep_train=feature_calculation(id_ep_train,non_natural_json_file)
# del id_ep_train
# id_ss_train = torch.load(base_path+'/tensor dataset/ss_train_id.pt')
# data_ss_train=feature_calculation(id_ss_train,non_natural_json_file)
# del id_ss_train,non_natural_json_file
# #合并三分支
# natural_train=[waveform_natural_train,image_natural_train,data_natural_train]
# del waveform_natural_train,image_natural_train
# ep_train=[waveform_ep_train,image_ep_train,data_ep_train]
# del waveform_ep_train,image_ep_train
# ss_train=[waveform_ss_train,image_ss_train,data_ss_train]
# del waveform_ss_train,image_ss_train
# print(natural_train[0].shape)
# print(natural_train[1].shape)
# print(natural_train[2].shape)
# #标签编码 0天然地震 1爆破 2塌陷
# zero_tensor = lable_making(natural_train,0)
# one_tensor = lable_making(ep_train,1)
# two_tensor= lable_making(ss_train,2)

