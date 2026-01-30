#这段 feature_calculation.py 里选中的部分代码的功能——它的用途是从 JSON 元数据中提取地震事件特征，为后续深度学习输入做准备，不过现在大部分功能是半成品或被注释掉的。
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from collections import Counter, OrderedDict
import random
import matplotlib
from torch.utils.data import DataLoader, TensorDataset
import obspy
from scipy import signal
#数据根路径(1,2,3)
dataset_id=1
#数据路径  
base_path='/home/zzy/Python projects/Dataset/'+'{}'.format(dataset_id)
j=0
#特征计算
def feature_calculation(id_list,json_file):
    data=[[]]
    #循环里可以加计算过程
    for i in id_list:
        j+=1
        information = json_file[i]
        print(information)
        # if 'mag' in information and information['mag']:
        #     data[0].append(float(information['mag']))
        # if 'se_mag' in information and information['se_mag']and information['se_mag'].replace('.', '', 1).isdigit():
        #     data[1].append(float(information['se_mag']))
        # if 'sn_mag' in information and information['sn_mag']and information['sn_mag'].replace('.', '', 1).isdigit():
        #     data[2].append(float(information['sn_mag']))
        if j >1:
            break
    # data=torch.tensor(data)
    # data = torch.transpose(data, 0, 1)
    # return data

id_ep_train = torch.load(base_path+'/tensor dataset/ep_train_id.pt')
non_natural_json_file = json.load(open('/home/zzy/Python projects/Dataset/DiTing2.0/CENC_DiTingv2_non_natural_earthquake.json', 'r'))
data_ep_train=feature_calculation(id_ep_train,non_natural_json_file)
