import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from torch.utils.data import random_split
from utils import feature_calculation,lable_making
import json
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

def mult_branch_loader(dataset_id, dataset_choose, base_path,batch_size):
    train_dataset=Mult_branch_loader(dataset_id, dataset_choose, base_path)
    #划分训练集与验证集
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader

def two_class_dataloader(natural_train_path_0,natural_train_path_1,ep_train_path,ss_train_path,
                batch_size=64,validation_percentage = 0.1):
    loaded_natural_train_0 = torch.load(natural_train_path_0)
    
    if natural_train_path_1:
        loaded_natural_train_1 = torch.load(natural_train_path_1)
        loaded_natural_train = torch.cat((loaded_natural_train_0, loaded_natural_train_1), dim=0)
        del loaded_natural_train_0,loaded_natural_train_1
    else:
        loaded_natural_train=loaded_natural_train_0
        del loaded_natural_train_0
    loaded_ep_train = torch.load(ep_train_path)
    loaded_ss_train = torch.load(ss_train_path)

    loaded_natural_train=torch.squeeze(loaded_natural_train)
    loaded_ep_train=torch.squeeze(loaded_ep_train)
    loaded_ss_train=torch.squeeze(loaded_ss_train)

    #划分训练集与验证集
    natural_train_size = loaded_natural_train.shape[0] - int(validation_percentage * loaded_natural_train.shape[0])
    ep_train_size = loaded_ep_train.shape[0] - int(validation_percentage * loaded_ep_train.shape[0])
    ss_train_size = loaded_ss_train.shape[0] - int(validation_percentage * loaded_ss_train.shape[0])

    natural_train=loaded_natural_train[:natural_train_size]
    ep_train=loaded_ep_train[:ep_train_size]
    ss_train=loaded_ss_train[:ss_train_size]
    nonnatural_train = torch.cat((ep_train, ss_train), dim=0)
    del ep_train,ss_train
    #标签编码 0天然地震 1非天然地震
    zero_tensor_train = lable_making(natural_train,0)
    one_tensor_train = lable_making(nonnatural_train,1)

    natural_valid=loaded_natural_train[natural_train_size:]
    ep_valid=loaded_ep_train[ep_train_size:]
    ss_valid=loaded_ss_train[ss_train_size:]
    nonnatural_valid= torch.cat((ep_valid, ss_valid), dim=0)
    del ep_valid,ss_valid
    #标签编码 0天然地震 1非天然地震
    zero_tensor_valid = lable_making(natural_valid,0)
    one_tensor_valid = lable_making(nonnatural_valid,1)

    del loaded_natural_train,loaded_ep_train,loaded_ss_train

    # 使用torch.cat在维度0上合并两个张量  
    train_tensor = torch.cat((natural_train,nonnatural_train), dim=0)
    del natural_train,nonnatural_train
    train_lable_tensor=torch.cat((zero_tensor_train, one_tensor_train), dim=0)
    del zero_tensor_train,one_tensor_train

    valid_tensor = torch.cat((natural_valid,nonnatural_valid), dim=0)
    del natural_valid,nonnatural_valid
    valid_lable_tensor=torch.cat((zero_tensor_valid, one_tensor_valid), dim=0)
    del zero_tensor_valid,one_tensor_valid

    # 创建TensorDataset并将图像数据和标签合并
    train_dataset = TensorDataset(train_tensor, train_lable_tensor)
    valid_dataset = TensorDataset(valid_tensor, valid_lable_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,valid_loader  

def two_class_test_dataloader(natural_test_path,ep_test_path,ss_test_path):
    loaded_natural_test = torch.load(natural_test_path)
    loaded_natural_test=torch.tensor(loaded_natural_test)
    loaded_ep_test = torch.load(ep_test_path)
    loaded_ep_test=torch.tensor(loaded_ep_test)
    loaded_ss_test = torch.load(ss_test_path)
    loaded_ss_test=torch.tensor(loaded_ss_test)
    loaded_nonnatural_test=torch.cat((loaded_ep_test, loaded_ss_test), dim=0)
    #标签编码 0天然地震 1爆破 2塌陷
    natural_label = lable_making(loaded_natural_test,0)
    nonnatural_label=lable_making(loaded_nonnatural_test,1)
    # print(loaded_natural_test.shape)
    # print(loaded_ep_test.shape)
    # print(loaded_ss_test.shape)
    # 使用torch.cat在维度0上合并两个张量  
    test_tensor = torch.cat((loaded_natural_test,loaded_nonnatural_test), dim=0)
    del loaded_natural_test,loaded_ep_test,loaded_ss_test
    test_lable_tensor=torch.cat((natural_label,nonnatural_label), dim=0)
    del natural_label,nonnatural_label
    # 创建TensorDataset并将图像数据和标签合并
    test_dataset = TensorDataset(test_tensor, test_lable_tensor)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    return test_loader

def noise_dataloader(natural_train_path_0,natural_train_path_1,ep_train_path,ss_train_path,noise_train_path,
                batch_size=64,validation_percentage = 0.1):
    loaded_natural_train_0 = torch.load(natural_train_path_0)
    if natural_train_path_1:
        loaded_natural_train_1 = torch.load(natural_train_path_1)
        loaded_natural_train = torch.cat((loaded_natural_train_0, loaded_natural_train_1), dim=0)
        del loaded_natural_train_0,loaded_natural_train_1
    else:
        loaded_natural_train=loaded_natural_train_0
        del loaded_natural_train_0
    loaded_ep_train = torch.load(ep_train_path)
    loaded_ss_train = torch.load(ss_train_path)
    loaded_noise_train = torch.load(noise_train_path)

    loaded_natural_train=torch.squeeze(loaded_natural_train)
    loaded_ep_train=torch.squeeze(loaded_ep_train)
    loaded_ss_train=torch.squeeze(loaded_ss_train)
    loaded_noise_train=torch.squeeze(loaded_noise_train)

    #划分训练集与验证集
    natural_train_size = loaded_natural_train.shape[0] - int(validation_percentage * loaded_natural_train.shape[0])
    ep_train_size = loaded_ep_train.shape[0] - int(validation_percentage * loaded_ep_train.shape[0])
    ss_train_size = loaded_ss_train.shape[0] - int(validation_percentage * loaded_ss_train.shape[0])
    noise_train_size = loaded_noise_train.shape[0] - int(validation_percentage * loaded_noise_train.shape[0])

    natural_train=loaded_natural_train[:natural_train_size]
    ep_train=loaded_ep_train[:ep_train_size]
    ss_train=loaded_ss_train[:ss_train_size]
    noise_train=loaded_noise_train[:noise_train_size]
    nonnatural_train = torch.cat((ep_train, ss_train), dim=0)
    
    #标签编码 0天然地震 1非天然地震 2噪声
    zero_tensor_train = lable_making(natural_train,0)
    one_ep_tensor_train = lable_making(ep_train,1)
    one_ss_tensor_train = lable_making(ss_train,1)
    two_tensor_train = lable_making(noise_train,2)
    one_tensor_train = torch.cat((one_ep_tensor_train, one_ss_tensor_train), dim=0)
    del one_ep_tensor_train,one_ss_tensor_train
    natural_valid=loaded_natural_train[natural_train_size:]
    ep_valid=loaded_ep_train[ep_train_size:]
    ss_valid=loaded_ss_train[ss_train_size:]
    noise_valid=loaded_noise_train[noise_train_size:]
    nonnatural_valid= torch.cat((ep_valid, ss_valid), dim=0)
    
    #标签编码 0天然地震 1非天然地震 2噪声
    zero_tensor_valid = lable_making(natural_valid,0)
    one_ep_tensor_valid = lable_making(ep_valid,1)
    one_ss_tensor_valid = lable_making(ss_valid,1)
    two_tensor_valid = lable_making(noise_valid,2)
    one_tensor_valid = torch.cat((one_ep_tensor_valid, one_ss_tensor_valid), dim=0)
    del loaded_natural_train,loaded_ep_train,loaded_ss_train,one_ep_tensor_valid,one_ss_tensor_valid

    # 使用torch.cat在维度0上合并两个张量  
    train_tensor = torch.cat((natural_train,nonnatural_train,noise_train), dim=0)
    del natural_train,nonnatural_train,noise_train
    train_lable_tensor=torch.cat((zero_tensor_train, one_tensor_train,two_tensor_train), dim=0)
    del zero_tensor_train,one_tensor_train,two_tensor_train

    valid_tensor = torch.cat((natural_valid,nonnatural_valid,noise_valid), dim=0)
    del natural_valid,nonnatural_valid,noise_valid
    valid_lable_tensor=torch.cat((zero_tensor_valid, one_tensor_valid,two_tensor_valid), dim=0)
    del zero_tensor_valid,one_tensor_valid,two_tensor_valid

    # 创建TensorDataset并将图像数据和标签合并
    train_dataset = TensorDataset(train_tensor, train_lable_tensor)
    valid_dataset = TensorDataset(valid_tensor, valid_lable_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,valid_loader  

def noise_test_dataloader(natural_test_path,ep_test_path,ss_test_path,noise_test_path):
    loaded_natural_test = torch.load(natural_test_path)
    loaded_natural_test=torch.tensor(loaded_natural_test)
    loaded_ep_test = torch.load(ep_test_path)
    loaded_ep_test=torch.tensor(loaded_ep_test)
    loaded_ss_test = torch.load(ss_test_path)
    loaded_ss_test=torch.tensor(loaded_ss_test)
    loaded_noise_test = torch.load(noise_test_path)
    loaded_noise_test=torch.tensor(loaded_noise_test)
    
    #标签编码 0天然地震 1非天然地震 2噪声
    natural_label = lable_making(loaded_natural_test,0)
    ep_label = lable_making(loaded_ep_test,1)
    ss_label= lable_making(loaded_ss_test,1)
    non_natural_label=torch.cat((ep_label, ss_label), dim=0)
    noise_label= lable_making(loaded_noise_test,2)
    
    # 使用torch.cat在维度0上合并两个张量  
    test_tensor = torch.cat((loaded_natural_test, loaded_ep_test,loaded_ss_test,loaded_noise_test), dim=0)
    del loaded_natural_test,loaded_ep_test,loaded_ss_test,loaded_noise_test
    test_lable_tensor=torch.cat((natural_label, non_natural_label,noise_label), dim=0)
    del natural_label,ep_label,ss_label,non_natural_label,noise_label
    # 创建TensorDataset并将图像数据和标签合并
    test_dataset = TensorDataset(test_tensor, test_lable_tensor)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    return test_loader

def dataloader(natural_train_path_0,natural_train_path_1,ep_train_path,ss_train_path,
                batch_size=64,validation_percentage = 0.1):
    loaded_natural_train_0 = torch.load(natural_train_path_0)
    # loaded_natural_train_0=torch.tensor(loaded_natural_train_0)
    if natural_train_path_1:
        loaded_natural_train_1 = torch.load(natural_train_path_1)
        # loaded_natural_train_1=torch.tensor(loaded_natural_train_1)
        loaded_natural_train = torch.cat((loaded_natural_train_0, loaded_natural_train_1), dim=0)
        del loaded_natural_train_0,loaded_natural_train_1
    else:
        loaded_natural_train=loaded_natural_train_0
        del loaded_natural_train_0
    loaded_ep_train = torch.load(ep_train_path)
    # loaded_ep_train=torch.tensor(loaded_ep_train)
    loaded_ss_train = torch.load(ss_train_path)

    # loaded_ss_train=torch.tensor(loaded_ss_train)
    
    loaded_natural_train=torch.squeeze(loaded_natural_train)
    loaded_ep_train=torch.squeeze(loaded_ep_train)
    loaded_ss_train=torch.squeeze(loaded_ss_train)
    #划分训练集与验证集
    natural_train_size = loaded_natural_train.shape[0] - int(validation_percentage * loaded_natural_train.shape[0])
    ep_train_size = loaded_ep_train.shape[0] - int(validation_percentage * loaded_ep_train.shape[0])
    ss_train_size = loaded_ss_train.shape[0] - int(validation_percentage * loaded_ss_train.shape[0])

    natural_train=loaded_natural_train[:natural_train_size]
    ep_train=loaded_ep_train[:ep_train_size]
    ss_train=loaded_ss_train[:ss_train_size]
    #标签编码 0天然地震 1爆破 2塌陷
    zero_tensor_train = lable_making(natural_train,0)
    one_tensor_train = lable_making(ep_train,1)
    two_tensor_train = lable_making(ss_train,2)

    natural_valid=loaded_natural_train[natural_train_size:]
    ep_valid=loaded_ep_train[ep_train_size:]
    ss_valid=loaded_ss_train[ss_train_size:]
    #标签编码 0天然地震 1爆破 2塌陷
    zero_tensor_valid = lable_making(natural_valid,0)
    one_tensor_valid = lable_making(ep_valid,1)
    two_tensor_valid= lable_making(ss_valid,2)
    del loaded_natural_train,loaded_ep_train,loaded_ss_train
    # 使用torch.cat在维度0上合并两个张量  
    train_tensor = torch.cat((natural_train, ep_train,ss_train), dim=0)
    del natural_train,ep_train,ss_train
    train_lable_tensor=torch.cat((zero_tensor_train, one_tensor_train,two_tensor_train), dim=0)
    del zero_tensor_train,one_tensor_train,two_tensor_train

    valid_tensor = torch.cat((natural_valid, ep_valid,ss_valid), dim=0)
    del natural_valid,ep_valid,ss_valid
    valid_lable_tensor=torch.cat((zero_tensor_valid, one_tensor_valid,two_tensor_valid), dim=0)
    del zero_tensor_valid,one_tensor_valid,two_tensor_valid

    # 创建TensorDataset并将图像数据和标签合并
    train_dataset = TensorDataset(train_tensor, train_lable_tensor)
    valid_dataset = TensorDataset(valid_tensor, valid_lable_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,valid_loader  
    
def test_dataloader(natural_test_path,ep_test_path,ss_test_path):
    loaded_natural_test = torch.load(natural_test_path)
    loaded_natural_test=torch.tensor(loaded_natural_test)
    loaded_ep_test = torch.load(ep_test_path)
    loaded_ep_test=torch.tensor(loaded_ep_test)
    loaded_ss_test = torch.load(ss_test_path)
    loaded_ss_test=torch.tensor(loaded_ss_test)

    #标签编码 0天然地震 1爆破 2塌陷
    natural_label = lable_making(loaded_natural_test,0)
    ep_label = lable_making(loaded_ep_test,1)
    ss_label= lable_making(loaded_ss_test,2)
    # print(loaded_natural_test.shape)
    # print(loaded_ep_test.shape)
    # print(loaded_ss_test.shape)
    # 使用torch.cat在维度0上合并两个张量  
    test_tensor = torch.cat((loaded_natural_test, loaded_ep_test,loaded_ss_test), dim=0)
    del loaded_natural_test,loaded_ep_test,loaded_ss_test
    test_lable_tensor=torch.cat((natural_label, ep_label,ss_label), dim=0)
    del natural_label,ep_label,ss_label
    # 创建TensorDataset并将图像数据和标签合并
    test_dataset = TensorDataset(test_tensor, test_lable_tensor)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    return test_loader


