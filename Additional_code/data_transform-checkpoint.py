import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import librosa
from sklearn.model_selection import StratifiedKFold
import os
import scipy

base_path='/root/autodl-tmp/Python projects/Dataset/expand'
def STFT(data, n_fft=100, hop_length=5,window='hann'):
    # 先将数据转换为 NumPy 数组，并保留实部
    data = data.numpy().real
    num_samples, num_features, num_time_steps = data.shape
    stft_data = np.zeros((num_samples, num_features, 1 + n_fft // 2,601), dtype=np.float32)
    data = np.array(data, dtype=np.float32)
    for i in range(num_samples):
        for k in range(num_features):
            # 计算STFT
            stft_data[i][k] = librosa.stft(data[i][k], n_fft=n_fft, 
                                           hop_length=hop_length,win_length=n_fft, 
                                           window=window)
    return stft_data

def stft_tensor_save(data):
    #将变换后的tensor保存
    STFT_data=[]
    for i in range(data.shape[0]):
        stft_data=data[i:i+1,:,:]
        stft_images = STFT(stft_data)
        STFT_data.append(stft_images)
    STFT_data=torch.tensor(STFT_data).squeeze(dim=1)
    return STFT_data

def MFCC(data, sample_rate=50,n_mfcc=13,n_fft=142, hop_length=42):
    data = np.array(data, dtype=np.float32)
    num_samples, num_features, num_time_steps = data.shape
    mfcc_data = np.zeros((num_samples, num_features, 40, 72), dtype=np.float32)
    
    for i in range(num_samples):
        for k in range(num_features):
            # 计算MFCC
            mfcc= librosa.feature.mfcc(y=data[i][k], sr=sample_rate,n_mfcc=n_mfcc,win_length=n_fft,hop_length=hop_length,n_fft=n_fft)
            #一阶差分
            delta_mfcc = librosa.feature.delta(data=mfcc)
            #二阶差分
            delta2_mfcc = librosa.feature.delta(data=mfcc, order=2)
            # 计算音频信号每帧的对数能量
            log_energy = librosa.amplitude_to_db(librosa.feature.rms(y=data[i][k],frame_length=n_fft, hop_length=hop_length))
    
            mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc,log_energy], axis=0)
            mfcc_data[i][k] = mfcc
            
    return mfcc_data

def mfcc_tensor_save(data):
    MFCC_data=[]
    for i in range(data.shape[0]):
        mfcc_data=data[i:i+1,:,:]
        mfcc_images = MFCC(mfcc_data)
        MFCC_data.append(mfcc_images)
    MFCC_data=torch.tensor(MFCC_data).squeeze(dim=1)
    return MFCC_data


loaded_natural_train = torch.load(base_path+'/natural_train_dataset.pt')
natural_train_STFT=stft_tensor_save(loaded_natural_train)
torch.save(natural_train_STFT, base_path+'/stft/natural_train_STFT.pt')
del natural_train_STFT
natural_train_MFCC=mfcc_tensor_save(loaded_natural_train)
torch.save(natural_train_MFCC, base_path+'/mfcc/natural_train_MFCC.pt')
del loaded_natural_train,natural_train_MFCC

loaded_natural_test = torch.load(base_path+'/natural_test_dataset.pt')
natural_test_STFT=stft_tensor_save(loaded_natural_test)
torch.save(natural_test_STFT, base_path+'/stft/natural_test_STFT.pt')
del natural_test_STFT
natural_test_MFCC=mfcc_tensor_save(loaded_natural_test)
torch.save(natural_test_MFCC, base_path+'/mfcc/natural_test_MFCC.pt')
del loaded_natural_test,natural_test_MFCC

loaded_ep_train = torch.load(base_path+'/ep_train_dataset.pt')
ep_train_STFT=stft_tensor_save(loaded_ep_train)
torch.save(ep_train_STFT, base_path+'/stft/ep_train_STFT.pt')
del ep_train_STFT
ep_train_MFCC=mfcc_tensor_save(loaded_ep_train)
torch.save(ep_train_MFCC, base_path+'/mfcc/ep_train_MFCC.pt')
del loaded_ep_train,ep_train_MFCC

loaded_ep_test = torch.load(base_path+'/ep_test_dataset.pt')
ep_test_STFT=stft_tensor_save(loaded_ep_test)
torch.save(ep_test_STFT, base_path+'/stft/ep_test_STFT.pt')
del ep_test_STFT
ep_test_MFCC=mfcc_tensor_save(loaded_ep_test)
torch.save(ep_test_MFCC, base_path+'/mfcc/ep_test_MFCC.pt')
del loaded_ep_test,ep_test_MFCC

loaded_ss_train = torch.load(base_path+'/ss_train_dataset.pt')
ss_train_STFT=stft_tensor_save(loaded_ss_train)
torch.save(ss_train_STFT, base_path+'/stft/ss_train_STFT.pt')
del ss_train_STFT
ss_train_MFCC=mfcc_tensor_save(loaded_ss_train)
torch.save(ss_train_MFCC, base_path+'/mfcc/ss_train_MFCC.pt')
del loaded_ss_train,ss_train_MFCC

loaded_ss_test = torch.load(base_path+'/ss_test_dataset.pt')
ss_test_STFT=stft_tensor_save(loaded_ss_test)
torch.save(ss_test_STFT, base_path+'/stft/ss_test_STFT.pt')
del ss_test_STFT
ss_test_MFCC=mfcc_tensor_save(loaded_ss_test)
torch.save(ss_test_MFCC, base_path+'/mfcc/ss_test_MFCC.pt')
del loaded_ss_test,ss_test_MFCC

loaded_noise_train = torch.load(base_path+'/noise_train.pt')
noise_train_STFT=stft_tensor_save(loaded_noise_train)
torch.save(noise_train_STFT, base_path+'/stft/noise_train_STFT.pt')
del noise_train_STFT
noise_train_MFCC=mfcc_tensor_save(loaded_noise_train)
torch.save(noise_train_MFCC, base_path+'/mfcc/noise_train_MFCC.pt')
del loaded_noise_train,noise_train_MFCC

loaded_noise_test = torch.load(base_path+'/noise_test.pt')
noise_test_STFT=stft_tensor_save(loaded_noise_test)
torch.save(noise_test_STFT, base_path+'/stft/noise_test_STFT.pt')
del noise_test_STFT
noise_test_MFCC=mfcc_tensor_save(loaded_noise_test)
torch.save(noise_test_MFCC, base_path+'/mfcc/noise_test_MFCC.pt')
del loaded_noise_test,noise_test_MFCC