import torch
import pandas as pd
import numpy as np
from functools import partial
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import GUI_ui
import utils
print("UTILS FILE =", utils.__file__)
import os
# 定义分类函数
def Test(ui):
    # 获取当前工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 输入文件路径：当前目录下的 'events.pt'
    input_path = os.path.join(current_dir, 'events.pt')
    
    if not os.path.exists(input_path):
        ui.show_1.setText("输入文件不存在！")
        return
    dataset_choose = 'MFCC_40_72'
    Num_classes = 3
    classes = ['natural','ep','ss']
    try:
        # 加载输入数据
        input_test = torch.load(input_path)
        tensor_test = torch.tensor(input_test, dtype=torch.float32)  # 确保类型正确
        # 获取数据的维度
        data_shape = tensor_test.shape
        # 显示维度信息
        ui.show_1.setText(f"成功加载数据！\n数据维度：{data_shape}")
        #数据预处理
        for i in range(tensor_test.size(0)):
            for j in range(tensor_test.size(1)):
                # 获取当前通道和样本的数据
                data = tensor_test[i, j, :].numpy()
                # 对数据进行去趋势处理
                detrended_data = utils.detrend(data)
                # 对去趋势后的数据进行1-25Hz滤波处理
                filtered_data = utils.filter(detrended_data,lowcut = 1,highcut = 20)
                # 归一化处理
                normalized_data = utils.z_score_normalize(filtered_data)
                # 将结果赋值回原始张量
                tensor_test[i, j, :] = torch.tensor(np.ascontiguousarray(normalized_data))
        #MFCC
        tensor_test = utils.mfcc_transform(tensor_test)

        # 定义模型
        model_choose = ui.comboBox.currentText()  # 确保从下拉框获取文本
        model = utils.get_network(dataset_choose=dataset_choose, net=model_choose, Num_classes=Num_classes)
        
        # 加载模型参数
        checkpoint_path = utils.checkpoint_choose(model_choose) # 获取模型权重文件路径
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()  # 设置为评估模式
        
        # 预测输出
        with torch.no_grad():  # 禁用梯度计算，加快推理速度
            output = model(tensor_test)
            values, preds = torch.max(output, 1)  # 获取每个样本的最大概率和对应类别索引
            
            # 映射索引到类别名称
            predicted_classes = [classes[pred] for pred in preds.tolist()]

        output_folder = ui.lineEdit_8.text()  # 获取输出文件路径
        # 检查路径是否存在或为空
        if not output_folder or not os.path.exists(output_folder):
            # 如果路径不存在或未指定，使用当前代码所在文件夹
            output_folder = os.path.dirname(os.path.abspath(__file__))
        # 确保路径为绝对路径
        output_folder = os.path.abspath(output_folder)
        # 获取文件名并将其中的 '/' 替换为 '?'
        filename = ui.lineEdit_1.text().replace('/', '_') + '.txt'
        # 拼接出完整的文件路径
        output_path = os.path.join(output_folder, filename)
        # 确保文件夹路径存在
        if not os.path.exists(output_folder):
            ui.show_1.setText(f"输出文件夹不存在：{output_folder}")
            return
        # 保存预测结果到文件
        try:
            with open(output_path, 'w') as f:
                for i, pred_class in enumerate(predicted_classes):
                    f.write(f"Sample {i + 1}: {pred_class}\n")
            # 将 predicted_classes 转换为字符串，并显示在 UI 上
            predicted_classes_str = "\n".join([f"Sample {i + 1}: {pred}" for i, pred in enumerate(predicted_classes)])
            ui.show_1.setText(f"预测结果已保存至 {output_path}\n\n预测类别：\n{predicted_classes_str}")
        except Exception as e:
            ui.show_1.setText(f"保存文件时出错：{str(e)}")
    except Exception as e:
        ui.show_1.setText(f"预测失败：{str(e)}")

if __name__ == '__main__':
    # 加载GUI界面
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    gui = GUI_ui.Ui_MainWindow()
    gui.setupUi(MainWindow)
    MainWindow.show()
    
    # 为按钮绑定相关功能函数完成功能添加
    gui.pushButton_7.clicked.connect(partial(Test, gui))
    sys.exit(app.exec_())
