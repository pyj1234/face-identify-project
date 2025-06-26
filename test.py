import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FaceRecognitionModel # 导入 PyTorch 模型
from dataset_utils import load_and_split_dataset # 导入自定义数据加载工具
import os

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集根目录 (应与训练时一致)
DATASET_ROOT_DIR = './dataset'
MODEL_PATH = 'face_recognition_model_pytorch.pth' # 训练好的模型路径

# 模型参数
IMG_HEIGHT, IMG_WIDTH = 112, 112
BATCH_SIZE = 32
TRAIN_SPLIT_RATIO = 0.7 # 需要与训练时的划分比例一致，才能正确加载对应的测试集

def test_model():
    """
    测试训练好的人脸识别模型 (PyTorch)，使用自定义 PGM 数据加载。
    """
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 未找到模型文件 {MODEL_PATH}。请先运行 train_pytorch.py 训练模型。")
        return

    # 加载模型状态字典和类别映射
    print(f"正在加载模型状态字典和类别映射从: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    class_mapping = checkpoint['class_to_idx']
    num_classes = len(class_mapping)
    print(f"模型加载成功，类别数: {num_classes}")

    # 构建模型实例并加载状态字典
    model = FaceRecognitionModel(num_classes=num_classes)
    model.load_state_dict(model_state_dict)
    model.to(device) # 确保模型在正确设备上
    model.eval() # 设置模型为评估模式

    # 加载数据集 (这里我们加载整个数据集，然后根据 load_and_split_dataset 的划分获取测试集)
    # 注意：load_and_split_dataset 会重新进行划分，确保使用相同的随机种子以获得相同的划分结果
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

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 定义损失函数 (测试时也可以计算损失)
    criterion = nn.CrossEntropyLoss()

    # 评估模型
    print("开始在测试集上评估模型...")
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total

    print(f"测试集 Loss: {avg_test_loss:.4f}")
    print(f"测试集 Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    # 确保您的数据集位于项目根目录下的 'dataset' 文件夹
    test_model()