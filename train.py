# train_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FaceRecognitionModel # 导入 PyTorch 模型
from dataset_utils import load_and_split_dataset # 导入自定义数据加载工具
import os
import time

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集根目录 (请根据您的实际情况修改)
DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
MODEL_SAVE_PATH = 'face_recognition_model_pytorch.pth' # 模型保存路径

# 模型和训练参数
IMG_HEIGHT, IMG_WIDTH = 112, 112
BATCH_SIZE = 32
EPOCHS = 20 # 训练轮数
TRAIN_SPLIT_RATIO = 0.7 # 训练集比例

def train_model():
    """
    训练人脸识别模型 (PyTorch)，使用自定义 PGM 数据加载。
    """
    # 加载并划分数据集
    print(f"正在加载数据集: {DATASET_ROOT_DIR}")
    train_dataset, validation_dataset, class_mapping = load_and_split_dataset(
        DATASET_ROOT_DIR,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )

    if train_dataset is None or validation_dataset is None:
        print("数据集加载失败，请检查路径和文件格式。")
        return

    num_classes = len(class_mapping)
    print(f"数据集加载成功，共有 {num_classes} 个类别。")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers 根据您的机器性能调整
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 构建模型并移动到设备
    model = FaceRecognitionModel(num_classes=num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss() # 交叉熵损失，适用于多分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 优化器

    # 训练循环
    print("开始训练模型...")
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印训练信息
            if (i + 1) % 50 == 0: # 每隔50个批次打印一次 (根据数据集大小调整)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}")


        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{EPOCHS}] finished in {epoch_time:.2f} seconds.")
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {running_loss/len(train_loader):.4f}")


        # 验证模型
        model.eval() # 设置模型为评估模式
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # 在验证/测试时禁用梯度计算
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) # 获取预测结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = correct / total

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型 (只保存状态字典)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # 保存模型状态字典和类别映射
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_mapping,
            }, MODEL_SAVE_PATH)
            print(f"保存了验证集准确率最高的模型状态和类别映射到: {MODEL_SAVE_PATH}")

    print("训练完成。")

if __name__ == '__main__':
    # 确保您的数据集位于项目根目录下的 'dataset' 文件夹
    # 结构应为: dataset/s1/*.pgm, dataset/s2/*.pgm, ...
    train_model()