import numpy as np
import scipy
import torch
from scipy.fftpack import dct
from scipy.io import wavfile
# 分帧窗口长度
WIN_LEN = 128
# 采样间隔
HOP_LEN = 125
# FFT个数
N_FFT = 512
# mel滤波器个数
N_FILT = 40
# 倒谱系数个数
NUM_CEPS = 13
# 音频采样率
sample_rate = 50


def pre_emphasised(data):
    """ 预加重
    :rtype: object
    """
    pre_emphasis = 0.96
    data = np.append([data[0]], [(data[i + 1] - pre_emphasis * data[i]) for i in range(len(data) - 1)])
    return data

def get_hann_window(length=255):
    """ hanning窗
    """
    window = np.hanning(length)
    window.shape = [1, -1]
    return window.astype(np.float32)

def get_frames(pcm, frame_len, hop_len):
    """ 分帧
    :rtype: [帧个数，帧长度]
    """
    pcm_len = len(pcm)

    frames_num = 1 + (pcm_len - frame_len) // hop_len
    frames_num = int(frames_num)
    frames = []
    for i in range(frames_num):
        s = i * hop_len
        e = s + frame_len
        if e > pcm_len:
            e = pcm_len
        frame = pcm[s: e]
        frame = np.pad(frame, (0, frame_len - len(frame)), 'constant')
        frame.shape = [1, -1]
        frames.append(frame)
    frames = np.concatenate(frames, axis=0)
    return frames

def stft(frames):
    """ 计算短时傅立叶变换和功率谱
    :param frames: 分帧后数据
    :return: 功率谱
    """
    # fft后的振幅
    mag_frames = np.absolute(np.fft.rfft(frames, N_FFT))
    # 功率谱
    pow_frames = ((1.0 / N_FFT) * ((mag_frames) ** 2))
    print("pow_frames", pow_frames.shape)
    return pow_frames

def get_filter_bank(pow_frames):
    """ 提取mel刻度和各频段对数能量值
    """
    low_freq_mel = 0
    # 频率转换为Mel尺度
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # 对mel线性分区
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N_FILT + 2)
    # Mel尺度上point转频率
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((N_FFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((N_FILT, int(np.floor(N_FFT / 2 + 1))))

    for m in range(1, N_FILT + 1):
        # left
        f_m_minus = int(bin[m - 1])
        # center
        f_m = int(bin[m])
        # right
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    print("pow_frames,fbank", pow_frames.shape,fbank.shape)
    # [num_frame,pow_frame] dot [num_filter, num_pow]
    # 每帧对数能量值在对应滤波器频段相乘累加
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # 能量取对数
    filter_banks = 20 * np.log10(filter_banks)
    print("filter_banks", filter_banks.shape)
    return filter_banks

def get_MFCCs(filter_banks):
    """ 获取最终MFCC系数
    :param filter_banks: 经过Mel滤波器的对数能量
    """
    # 对数能量带入离散余弦变换公式
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (NUM_CEPS + 1)]
    (nframes, ncoeff) = mfcc.shape
    print("mfcc.shape", mfcc.shape)


if __name__ == '__main__':
    base_path='/root/autodl-tmp/Dataset/1'
    # 读取音频
    data = torch.load(base_path+'/tensor dataset/natural_train_dataset.pt')
    # 预加重
    data = pre_emphasised(np.array(data))
    # 获取汉宁窗
    _han = get_hann_window(length=WIN_LEN)
    # 分帧
    frames = get_frames(data, WIN_LEN, HOP_LEN)
    # 加窗
    frames = frames*_han
    # 傅立叶变换+得到功率谱
    pow_frames = stft(frames)
    # mel滤波器获取mel对数功率谱
    filter_banks = get_filter_bank(pow_frames)
    # 离散余弦变换，获取mel频谱倒谱系数
    get_MFCCs(filter_banks)