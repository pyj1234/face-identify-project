# test_recognition.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model1 import FaceRecognitionONet # 导入新定义的模型
from dataset_utils import load_and_split_dataset # 导入自定义数据加载工具
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集根目录 (应与训练时一致)
DATASET_ROOT_DIR = './dataset'
MODEL_PATH = 'face_recognition_onet.pth' # 训练好的模型路径

# 模型参数 (应与训练时一致)
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 32
TRAIN_SPLIT_RATIO = 0.3 # 需要与训练时的划分比例一致

def test_model():
    """
    测试训练好的人脸识别模型。
    """
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 未找到模型文件 {MODEL_PATH}。请先运行 train_recognition.py 训练模型。")
        return

    # 加载模型状态字典和类别映射
    print(f"正在加载模型状态字典和类别映射从: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        class_mapping = checkpoint['class_to_idx']
        num_classes = len(class_mapping)
        print(f"模型加载成功，类别数: {num_classes}")
        class_labels = list(class_mapping.keys()) # 获取类别名称列表

    except Exception as e:
        print(f"错误: 加载模型或类别映射失败 - {e}")
        return

    # 构建模型实例并加载状态字典
    model = FaceRecognitionONet(num_classes=num_classes)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval() # 设置模型为评估模式
    print("模型初始化并加载状态成功。")

    # 加载数据集 (获取测试数据集)
    print(f"正在加载数据集用于测试: {DATASET_ROOT_DIR}")
    # 调用 load_and_split_dataset 来获取测试数据集对象
    # load_and_split_dataset 内部会处理好图片的加载和转换
    _, test_dataset, _ = load_and_split_dataset(
        DATASET_ROOT_DIR,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )

    if test_dataset is None:
         print("测试数据集加载失败。")
         return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) # num_workers 保持与训练一致

    # 收集真实标签和预测结果
    true_labels = []
    predicted_labels = []

    print("开始在测试集上进行预测...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # 获取 logits
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果 (类别索引)

            true_labels.extend(labels.cpu().numpy()) # 将 Tensor 转为 NumPy 数组
            predicted_labels.extend(predicted.cpu().numpy())

    # 计算评估指标
    print("\n--- 测试结果 ---")

    # 准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"准确率 (Accuracy): {accuracy:.4f}")

    # 分类报告 (包含精确率、召回率、F1分数)
    print("\n分类报告 (Classification Report):")
    # target_names 的顺序需要与类别索引顺序一致
    print(classification_report(true_labels, predicted_labels, target_names=class_labels))

    # 混淆矩阵
    print("\n混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)

if __name__ == '__main__':
    # 在 Windows 下使用多进程数据加载，建议将测试函数放在 if __name__ == '__main__': 保护块内
    test_model()