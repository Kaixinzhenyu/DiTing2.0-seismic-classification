import torch
import torch.nn as nn
class VGG19(torch.nn.Module):
    def __init__(self,In_channel=3,num_class=2):
        super(VGG19, self).__init__()
        self.feature = torch.nn.Sequential(

            torch.nn.Conv1d(In_channel, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.AdaptiveAvgPool1d(7)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(3584,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(-1, 3584)
        # print(x.shape)
        x = self.classifer(x)
        return x

def vgg19(dataset_choose,num_classes=2):
    return VGG19(num_class=num_classes)


