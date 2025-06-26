import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model1 import FaceRecognitionONet # 导入新定义的模型
from dataset_utils import load_and_split_dataset # 导入自定义数据加载工具
import os
import time

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
# if device.type == 'cuda':
#     print("GPU设备可用")
#     print("GPU数量: " + str(torch.cuda.device_count()))
#     print("GPU名称: " + torch.cuda.get_device_name(0))

# 数据集根目录
DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
MODEL_SAVE_PATH = 'face_recognition_onet.pth' # 模型保存路径

# 模型和训练参数
IMG_HEIGHT, IMG_WIDTH = 48, 48 # ONet 输入尺寸
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_SPLIT_RATIO = 0.3 # 训练集比例

def train_model():
    """
    训练人脸识别模型 (基于 ONet 特征提取部分)。
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
    # 建议 num_workers > 0 以加速数据加载，但在 Windows 上可能需要放在 if __name__ == '__main__': 保护块内运行
    # 如果遇到多进程问题，请尝试 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 构建模型并移动到设备
    model = FaceRecognitionONet(num_classes=num_classes).to(device)

    # 损失函数和优化器
    # 对于多分类任务，使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 优化器

    # 如果存在之前保存的模型，加载继续训练
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"正在加载模型状态字典从: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 如果保存了 optimizer 状态，也可以加载
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("模型状态字典加载成功，继续训练...")
    else:
        print("未找到模型文件，从头开始训练...")

    # 训练循环
    print("开始训练模型...")
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
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

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            print("预测结果:", predicted)
            print("真实标签:", labels)
            correct += (predicted == labels).sum().item()

            # 打印训练信息 (每个批次或每隔N个批次)
            if (i + 1) % 50 == 0: # 每隔50个批次打印一次 (根据数据集大小调整)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}")


        epoch_time = time.time() - start_time
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] finished in {epoch_time:.2f} seconds.")
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # # 验证模型
        # model.eval() # 设置模型为评估模式
        # val_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad(): # 在验证/测试时禁用梯度计算
        #     for inputs, labels in validation_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)

        #         val_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1) # 获取预测结果
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # avg_val_loss = val_loss / len(validation_loader)
        # val_accuracy = correct / total

        # print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # # 保存最佳模型 (只保存状态字典和类别映射)
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        #     # 保存模型状态字典和类别映射
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'class_to_idx': class_mapping,
        #         # 'optimizer_state_dict': optimizer.state_dict(), # 如果需要保存优化器状态以便继续训练
        #     }, MODEL_SAVE_PATH)
        #     print(f"保存了验证集准确率最高的模型状态和类别映射到: {MODEL_SAVE_PATH}")

        # 保存最佳模型 (只保存状态字典和类别映射)
        if train_accuracy >= best_val_accuracy:
            best_val_accuracy = train_accuracy
            # 保存模型状态字典和类别映射
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': class_mapping,
                # 'optimizer_state_dict': optimizer.state_dict(), # 如果需要保存优化器状态以便继续训练
            }, MODEL_SAVE_PATH)
            print(f"保存了训练集准确率最高的模型状态和类别映射到: {MODEL_SAVE_PATH}")

    print("训练完成。")

if __name__ == '__main__':
    # 确保您的数据集位于项目根目录下的 'dataset' 文件夹
    # 结构应为: dataset/s1/*.pgm, dataset/s2/*.pgm, ...
    train_model()