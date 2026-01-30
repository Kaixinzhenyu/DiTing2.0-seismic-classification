import obspy
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os

# 设置文件路径，假设 SAC 文件位于 'test_data' 文件夹中
file_path = './test_data/'

windows = 5    # 切割窗口长度（秒）
THRESH_ON = 8   # STA/LTA 触发开启阈值
THRESH_OFF = 5  # STA/LTA 触发关闭阈值
SHORT_WINDOW = 0.1  # STA 窗口长度（秒）
LONG_WINDOW = 1     # LTA 窗口长度（秒）

# 获取 SAC 文件列表
sac_files = glob.glob(os.path.join(file_path, '*.sac'))

# 用于存储所有事件片段的列表
all_events_data = []

for file in sac_files:
    # 读取 SAC 文件
    input_trace = obspy.read(file)[0]
    sampling_interval = input_trace.stats.delta
    sampling_frequency = int(1 / sampling_interval)
    total_point = input_trace.stats.npts
    total_time = total_point * sampling_interval
    total_t = np.arange(0, total_time, sampling_interval)
    
    # 计算 STA/LTA 特征函数
    cft = classic_sta_lta(input_trace.data, int(SHORT_WINDOW / sampling_interval), int(LONG_WINDOW / sampling_interval))
    
    # 检测触发器
    cft_triggers = trigger_onset(cft, THRESH_ON, THRESH_OFF)
    
    # 处理每个检测到的事件
    for idx, trigger in enumerate(cft_triggers):
        event_start_sample = trigger[0]
        event_end_sample = trigger[1]
        event_duration = (event_end_sample - event_start_sample) * sampling_interval  # 事件持续时间（秒）
        
        # 计算事件前后时间
        pre_event_time = (windows - event_duration) / 3
        post_event_time = windows - event_duration - pre_event_time
        
        # 计算裁剪的开始和结束时间
        start_time = input_trace.stats.starttime + event_start_sample * sampling_interval - pre_event_time
        end_time = start_time + windows
    
        # 处理边界情况
        if start_time < input_trace.stats.starttime:
            start_time = input_trace.stats.starttime
            end_time = start_time + windows
        if end_time > input_trace.stats.endtime:
            end_time = input_trace.stats.endtime
            start_time = end_time - windows
    
        # 裁剪波形
        trimmed_trace = input_trace.copy()
        trimmed_trace.trim(starttime=start_time, endtime=end_time, pad=True, fill_value=0)
    
        # 确保裁剪后的数据长度正确
        expected_points = int(windows * sampling_frequency)
        actual_points = len(trimmed_trace.data)
        if actual_points < expected_points:
            # 如有必要，填充数据
            padding = np.zeros(expected_points - actual_points)
            trimmed_trace.data = np.concatenate((trimmed_trace.data, padding))
        elif actual_points > expected_points:
            # 如有必要，截断数据
            trimmed_trace.data = trimmed_trace.data[:expected_points]
        
        # 将裁剪后的数据添加到事件列表中
        event_tensor = torch.tensor(trimmed_trace.data, dtype=torch.float32)
        all_events_data.append(event_tensor)

# 将所有事件片段堆叠为一个多维 Tensor
all_events_tensor = torch.stack(all_events_data)

# 保存为一个 .pt 文件
output_filename = os.path.join(file_path, 'all_events.pt')
torch.save(all_events_tensor, output_filename)

print("所有事件片段已保存为单个 Tensor 文件：all_events.pt")






























