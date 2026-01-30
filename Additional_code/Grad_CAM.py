import torch
import numpy as np
from utils import get_network
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import os
import random
import librosa
import librosa.display
from PIL import Image

# 数据根路径(3,expand)
dataset_id = 'expand'
dataset_choose = 'STFT_72_72'#STFT_72_72 MFCC_40_72
model_choose = 'resnet18'
class_choose = 'ss'  # 'natural', 'ep', 'ss'
base_path = '/root/autodl-tmp/Python projects/Dataset/' + f'{dataset_id}'
checkpoint_basic_path = '/root/autodl-tmp/Python projects/my_project/checkpoint'
checkpoint_name = 'model_best_stft_resnet18_32_0.0001_0.0001_3_5'
checkpoint_path = f"{checkpoint_basic_path}/{dataset_id}/{checkpoint_name}.pth"

# 创建输出文件夹
output_dir = '/root/autodl-tmp/cam_outputs/' + dataset_choose
os.makedirs(output_dir, exist_ok=True)

# 加载模型权重，映射到 CPU
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
Num_classes = 3
classes = ['natural', 'ep', 'ss'] if Num_classes == 3 else ['natural', 'nonnatural']

model = get_network(dataset_choose, model_choose, Num_classes)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 加载数据
if dataset_id == '3':
    if dataset_choose == '1d':
        data_path = base_path + '/tensor dataset/natural_test_dataset.pt'
    elif dataset_choose == 'STFT_72_72':
        data_path = base_path + '/processed_data/' + dataset_choose + '/natural_test_STFT.pt'
    elif dataset_choose == 'MFCC_40_72':
        data_path = base_path + '/processed_data/' + dataset_choose + '/natural_test_MFCC.pt'
elif dataset_id == 'expand':
    if dataset_choose == '1d':
        data_path = base_path + '/natural_test_dataset.pt'
    elif dataset_choose == 'STFT_72_72':
        data_path = base_path + '/stft/natural_test_STFT.pt'
    elif dataset_choose == 'MFCC_40_72':
        data_path = base_path + '/mfcc/natural_test_MFCC.pt'
input_tensor = torch.load(data_path)

# 随机选取样本
num_samples = 5
total_samples = len(input_tensor)
if total_samples < num_samples:
    num_samples = total_samples
sample_indices = random.sample(range(total_samples), num_samples)

# 定义目标层
if model_choose == 'resnet18':
    target_layers = [model.conv5_x[-1]]

# 初始化 GradCAM
cam = GradCAM(model=model, target_layers=target_layers)

# 对于每个样本，计算 CAM 并保存
for idx in sample_indices:
    sample = input_tensor[idx]
    rgb_img = sample[0]
    if dataset_choose =='STFT_72_72':
        # 将STFT数据可视化
        plt.figure(figsize=(12, 4))
        stft_display = librosa.amplitude_to_db(np.abs(rgb_img.numpy()), ref=np.max)
        librosa.display.specshow(stft_display, sr=50, hop_length=3, x_axis="s", y_axis="hz")
        plt.axis('off')
        output_path = os.path.join(output_dir, f'origin_{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
        plt.savefig(output_path, format="jpg", bbox_inches="tight", pad_inches=0)
        plt.close()
    elif dataset_choose =='MFCC_40_72':
        # 将MFCC数据可视化
        plt.figure(figsize=(12, 6))
        # 将 PyTorch 张量转换为 NumPy 数组
        librosa.display.specshow(rgb_img.numpy())
        plt.axis('off')
        output_path = os.path.join(output_dir, f'origin_{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
        plt.savefig(output_path, format="jpg", bbox_inches="tight", pad_inches=0)
        plt.close()

    # 读取保存的 STFT 图像并裁剪空白区域
    image = cv2.imread(output_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    image_cropped = image[y:y+h, x:x+w]

    # # 保存裁剪后的图像
    # output_path_cropped = os.path.join(output_dir, f'cropped_{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
    # cv2.imwrite(output_path_cropped, image_cropped)

    # 转换为张量并叠加 Grad-CAM
    stft_tensor = torch.from_numpy(image_cropped).permute(2, 0, 1).float() / 255.0
    target_tensor = sample.unsqueeze(0).to(torch.float32)
    targets = [ClassifierOutputTarget(0)]

    # 计算 Grad-CAM
    grayscale_cam = cam(input_tensor=target_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (stft_tensor.shape[2], stft_tensor.shape[1]))

    # 叠加 Grad-CAM
    cam_overlay = show_cam_on_image(stft_tensor.permute(1, 2, 0).numpy(), grayscale_cam_resized, use_rgb=True)

    # 保存叠加后的图像
    overlay_output_path = os.path.join(output_dir, f'overlay_{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
    cv2.imwrite(overlay_output_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))

print(f"已将 {num_samples} 个样本的原始 STFT 图像和叠加的 Grad-CAM 结果保存到文件夹 '{output_dir}' 中。")

# import torch
# import numpy as np
# from utils import get_network
# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import cv2
# import os
# import random
# import librosa
# # 数据根路径(3,expand)
# dataset_id = '3'
# # 数据集选择 ('1d', 'GADF', 'STFT_72_72', 'MFCC_40_72')
# dataset_choose = 'STFT_72_72'
# # 模型选择
# model_choose = 'resnet18'
# class_choose = 'natural'  # 'natural', 'ep', 'ss'
# # 数据路径
# base_path = '/root/autodl-tmp/Python projects/Dataset/' + f'{dataset_id}'
# # 模型权重保存的路径
# checkpoint_basic_path = '/root/autodl-tmp/Python projects/my_project/checkpoint'
# checkpoint_name = 'model_best_STFT_72_72_resnet18_32_0.0001_0.0001_3_0'
# checkpoint_path = f"{checkpoint_basic_path}/{dataset_id}/{checkpoint_name}.pth"

# # 创建输出文件夹
# output_dir = '/root/autodl-tmp/cam_outputs/' + dataset_choose
# os.makedirs(output_dir, exist_ok=True)

# # 加载模型权重，映射到 CPU
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# # 类别数
# Num_classes = 3
# classes = ['natural', 'ep', 'ss'] if Num_classes == 3 else ['natural', 'nonnatural']

# # 加载模型
# model = get_network(dataset_choose, model_choose, Num_classes)
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()

# # 加载数据
# if dataset_id == '3':
#     if dataset_choose == '1d':
#         data_path = base_path + '/tensor dataset/natural_test_dataset.pt'
#     elif dataset_choose == 'STFT_72_72':
#         if class_choose == 'natural':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/natural_test_STFT.pt'
#         elif class_choose == 'ep':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/ep_test_STFT.pt'
#         elif class_choose == 'ss':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/ss_test_STFT.pt'
#     elif dataset_choose == 'MFCC_40_72':
#         if class_choose == 'natural':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/natural_test_MFCC.pt'
#         elif class_choose == 'ep':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/ep_test_MFCC.pt'
#         elif class_choose == 'ss':
#             data_path = base_path + '/processed_data/' + dataset_choose + '/ss_test_MFCC.pt'
# elif dataset_id == 'expand':
#     # 若 dataset_id 是 expand，则定义不同的路径
#     ...

# input_tensor = torch.load(data_path)

# # 随机选取十个样本
# num_samples = 5
# total_samples = len(input_tensor)
# if total_samples < num_samples:
#     num_samples = total_samples
# sample_indices = random.sample(range(total_samples), num_samples)

# # 定义目标层
# if model_choose == 'resnet18':
#     target_layers = [model.conv5_x[-1]]
# elif model_choose == 'res18_capsnet':
#     target_layers = [model.capsLayer]

# # 初始化 GradCAM
# cam = GradCAM(model=model, target_layers=target_layers)

# # 对于每个样本，计算 CAM 并保存
# for idx in sample_indices:
#     # 取出样本
#     sample = input_tensor[idx]
#     rgb_img = sample[0]
#     #可视化
#     plt.figure(figsize=(12, 4))
#     stft_display= librosa.amplitude_to_db(np.abs(rgb_img.numpy()), ref=np.max)
#     librosa.display.specshow(stft_display, sr=50,hop_length=3,x_axis="s", y_axis="hz")  
#     output_path = os.path.join(output_dir, f'origin_{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
#     plt.savefig(output_path, format="jpg", bbox_inches="tight", pad_inches=0.1)
#     plt.tight_layout()
    
#     # 检查是否为单通道图像并扩展为伪 RGB
#     if len(rgb_img.shape) == 2:
#         rgb_img = np.stack([rgb_img] * 3, axis=-1)
    
#     # 归一化
#     rgb_img = rgb_img / np.max(rgb_img)

#     # 准备输入张量
#     target_tensor = sample.unsqueeze(0).to(torch.float32)

#     # 定义目标类别（可根据需要修改）
#     targets = [ClassifierOutputTarget(0)]

#     # 计算 CAM
#     grayscale_cam = cam(input_tensor=target_tensor, targets=targets)
#     grayscale_cam = grayscale_cam[0, :]

#     # 生成可视化结果
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#     # 保存可视化结果
#     visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
#     output_path = os.path.join(output_dir, f'{dataset_id}_{dataset_choose}_{model_choose}_{class_choose}_{idx}.jpg')
#     cv2.imwrite(output_path, visualization_bgr)

# print(f"已将 {num_samples} 个样本的原始图像和 Grad-CAM 可视化结果合成并保存到文件夹 '{output_dir}' 中。")


# if model_choose == 'gra_capsnet':
#     # 定义hook函数
#     def hook_fn(module, input, output):
#         model.att = output 
#     # 选择要注册hook的层
#     layer = model.digitCaps
#     # 注册hook函数
#     hook_handle = layer.register_forward_hook(hook_fn)
#     # 假设有输入数据
#     input_data = torch.randn(1, 3, 36, 72)

#     # 使用模型进行前向传播
#     output = model(input_data)

#     # 获取att
#     att_value = model.att
#     print(att_value.shape)
#     E = att_value.sum(dim=1)/att_value.size()[2]
#     print(E.shape)
#     # 移除hook函数
#     hook_handle.remove()
        
# else:
    # #归一化
    # rgb_img= input_tensor[0]
    # rgb_img = rgb_img.permute(1, 2, 0)
    # rgb_img = rgb_img.numpy().astype(np.float32)
    # rgb_img /= np.max(rgb_img)
    # # #画图
    # # fig ,ax = plt.subplots()
    # # ax.imshow(rgb_img)
    # # plt.savefig('output7.jpg', format='jpg')

    # target_tensor = input_tensor[0].unsqueeze(0).cuda().to(torch.float32)

    # if model_choose =='resnet18':
    #     target_layers = [model.conv5_x[-1]]
    # elif model_choose =='res18_capsnet':
    #     target_layers = [model.capsLayer]

    # cam = GradCAM(model=model, target_layers=target_layers)

    # targets = [ClassifierOutputTarget(0)]

    # grayscale_cam = cam(input_tensor=target_tensor, targets=targets)
    # print(grayscale_cam.shape)
    # # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]

    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # # 保存可视化结果
    # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    # cv2.imwrite('cam.jpg', visualization)