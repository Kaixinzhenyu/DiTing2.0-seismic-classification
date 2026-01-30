from torch import nn
import torch
class CNN1d(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        # 创建一个虚拟输入以获取期望的特征图大小
        dummy_input = torch.randn(1, 3, 3000)
        dummy_output = self.conv(dummy_input)
        dummy_output_size = dummy_output.view(dummy_output.size(0), -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(dummy_output_size, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 256), 
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def cnn1d(num_classes=3):
    return CNN1d(num_classes=num_classes)











