import numpy as np
from scipy import signal
import librosa
import torch
# 去趋势
def detrend(data):
    detrended_data = signal.detrend(data, axis=-1)
    return detrended_data

# 滤波  采样率50hz  1-20hz带通滤波
def filter(data, fs=50, lowcut=1, highcut=20):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')#带通
    #b, a = signal.butter(4, high, 'lowpass')#低通
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data

#使用 Z-Score 对输入数据进行标准化
def z_score_normalize(data):
    
    mean_value = np.mean(data)
    std_dev = np.std(data)
    
    # 检查标准差是否为零，避免除以零的情况
    if std_dev == 0:
        # 如果标准差为零，直接返回原始数据
        return data
    
    normalized_data = (data - mean_value) / std_dev
    return normalized_data

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

def mfcc_transform(data):
    MFCC_data=[]
    for i in range(data.shape[0]):
        mfcc_data=data[i:i+1,:,:]
        mfcc_images = MFCC(mfcc_data)
        MFCC_data.append(mfcc_images)
    MFCC_data=torch.tensor(MFCC_data).squeeze(dim=1)
    return MFCC_data

def get_network(dataset_choose,net,Num_classes=3,Reconstruct=False):
    if net == 'AlexNet':
        from models.alexnet import alexnet
        model = alexnet(dataset_choose,num_classes=Num_classes)
    elif net == 'VGG11':
        from models.vgg import vgg11_bn
        model = vgg11_bn(dataset_choose,num_classes=Num_classes)
    elif net == 'GoogleNet':
        from models.googlenet import googlenet
        model = googlenet(num_classes=Num_classes)
    elif net == 'ResNet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=Num_classes)
    elif net=="VIT":
        from models.vit import ViT
        model = ViT(image_height=40, image_width=72, patch_height=10, patch_width=12,
        num_classes = 3,dim =256,depth = 6,heads = 16,mlp_dim = 256,dropout = 0.1,
        emb_dropout = 0.1)
    
    elif net=="CCT":
        from models.cct import CompactTransformer
        model = CompactTransformer(image_height=40, image_width=72, 
                                    patch_height=10, patch_width=12, 
                                    num_classes=Num_classes, conv_embed=True)
   
    elif net == 'CapsNet':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose='cnn_capsnet')
    elif net == 'CapsNet+Res':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose='res_capsnet')
    elif net == 'CapsNet+Res+SE':
        from models.cnn_capsnet import cnn_capsnet
        model = cnn_capsnet(num_classes=Num_classes,model_choose='seres_capsnet')
    elif net == 'CapsNet+CCT':
        from models.cct_capsnet import cct_capsnet
        model = cct_capsnet(num_classes=Num_classes)
    return model

def checkpoint_choose(model_choose):
    base_path='/Users/peizhenyu/Desktop/Seismic Event Classification System/code/GUI/checkpoint/'
    if model_choose == 'AlexNet':
        checkpoint_path=base_path+'model_best_mfcc_alexnet_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'VGG11':
        checkpoint_path=base_path+'model_best_mfcc_vgg11_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'GoogleNet':
        checkpoint_path=base_path+'model_best_mfcc_googlenet_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'ResNet18':
        checkpoint_path=base_path+'model_best_mfcc_resnet18_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'VIT':
        checkpoint_path=base_path+'model_best_mfcc_vit_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'CCT':
        checkpoint_path=base_path+'model_best_mfcc_cct_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'CapsNet':
        checkpoint_path=base_path+'model_best_mfcc_cnn_capsnet_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'CapsNet+Res':
        checkpoint_path=base_path+'model_best_mfcc_res_capsnet_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'CapsNet+Res+SE':
        checkpoint_path=base_path+'model_best_mfcc_seres_capsnet_32_0.0001_0.0001_3_5'+'.pth'
    elif model_choose == 'CapsNet+CCT':
        checkpoint_path=base_path+'model_best_mfcc_cct_capsnet_32_0.0001_0.0001_3_44'+'.pth'
    return  checkpoint_path               
