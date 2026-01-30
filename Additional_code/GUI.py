import tkinter as tk
from tkinter import filedialog, messagebox
import obspy
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 加载预训练的残差卷积网络模型
class ResidualCNN(nn.Module):
    def __init__(self):
        super(ResidualCNN, self).__init__()
        # 定义一个简单的残差网络模型，具体结构可根据实际模型更改
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, 2)  # 假设是二分类
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = torch.relu(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out

# 创建主界面
class EarthquakeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAC地震事件分类")
        
        # 选择文件按钮
        self.file_button = tk.Button(root, text="选择SAC文件", command=self.load_file)
        self.file_button.pack()
        
        # 显示分类结果按钮
        self.result_button = tk.Button(root, text="显示分类结果", command=self.show_results)
        self.result_button.pack()
        
        # 图像显示区域
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack()
        
        # 初始化模型和数据
        self.model = ResidualCNN()
        self.model.load_state_dict(torch.load("pretrained_resnet_weights.pt"))
        self.model.eval()
        self.events_data = []
        self.classification_results = []
    
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("SAC files", "*.sac")])
        if not file_path:
            return
        
        # 读取 SAC 文件并提取事件片段
        self.extract_events(file_path)
        messagebox.showinfo("完成", "地震事件片段提取完成")

    def extract_events(self, file_path):
        input_trace = obspy.read(file_path)[0]
        sampling_interval = input_trace.stats.delta
        SHORT_WINDOW, LONG_WINDOW = 0.1, 1
        THRESH_ON, THRESH_OFF = 8, 5
        windows = 5
        
        # 计算 STA/LTA
        cft = classic_sta_lta(input_trace.data, int(SHORT_WINDOW / sampling_interval), int(LONG_WINDOW / sampling_interval))
        cft_triggers = trigger_onset(cft, THRESH_ON, THRESH_OFF)
        
        # 提取事件片段
        self.events_data.clear()
        for trigger in cft_triggers:
            event_start_sample = trigger[0]
            event_end_sample = trigger[1]
            start_time = input_trace.stats.starttime + event_start_sample * sampling_interval
            end_time = start_time + windows
            
            trimmed_trace = input_trace.copy()
            trimmed_trace.trim(starttime=start_time, endtime=end_time, pad=True, fill_value=0)
            
            # 确保数据长度
            expected_points = int(windows / sampling_interval)
            data = np.pad(trimmed_trace.data, (0, max(0, expected_points - len(trimmed_trace.data))), 'constant')[:expected_points]
            self.events_data.append(data)
    
    def classify_events(self):
        # 转换为Tensor并分类
        self.classification_results.clear()
        total_events = len(self.events_data)
        event_tensor = torch.tensor(self.events_data, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            outputs = self.model(event_tensor)
            _, predictions = torch.max(outputs, 1)
            self.classification_results = predictions.tolist()
        
        # 计算分类准确率
        correct_predictions = sum([1 for pred in self.classification_results if pred == 1])  # 假设类1为地震
        accuracy = correct_predictions / total_events * 100
        return accuracy
    
    def show_results(self):
        if not self.events_data:
            messagebox.showwarning("警告", "请先加载SAC文件")
            return
        
        accuracy = self.classify_events()
        event_counts = {0: self.classification_results.count(0), 1: self.classification_results.count(1)}
        
        # 显示结果
        messagebox.showinfo("分类结果", f"分类准确率: {accuracy:.2f}%\n"
                                         f"地震事件: {event_counts[1]} 个\n"
                                         f"非地震事件: {event_counts[0]} 个")
        
        # 更新绘图
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[0].plot(self.events_data[0], 'k', linewidth=1)  # 显示第一个事件
        self.axs[0].set_title("第一个事件片段波形")
        self.axs[1].bar(["非地震", "地震"], [event_counts[0], event_counts[1]], color=['blue', 'red'])
        self.axs[1].set_title("事件分类统计")
        self.canvas.draw()

# 创建 Tkinter 根窗口并运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = EarthquakeGUI(root)
    root.mainloop()




































