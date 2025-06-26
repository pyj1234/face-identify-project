from Res2Net_v1b import res2net50_v1b_26w_4s
from model2 import FaceIDResNetV1
from dataset_utils1 import FaceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 数据集根目录
DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
MODEL_DIR = 'param' # 模型参数保存目录
MODEL_SAVE_PATH = MODEL_DIR + '/face_recognition_onet.pth' # 模型保存路径

# 模型和训练参数
IMG_HEIGHT, IMG_WIDTH = 48, 48 # ONet 输入尺寸
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_SPLIT_RATIO = 0.3 # 训练集比例
MODEL_TYPE_NUM = 40 # 模型类别数
def plot_confusion_matrix(cm, class_names):
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    """
    绘制带统计信息的混淆矩阵
    parameters:
        cm: 混淆矩阵
        class_names: 类别名称列表（如['s1','s2',...]）
    """
    # 创建画布
    plt.figure(figsize=(12, 10))
    
    # 使用Seaborn绘制热力图
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
    
    # 添加标签和标题
    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=16)
    
    # 调整标签显示
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 显示图形
    plt.tight_layout()
    plt.show()

def test(model, test_loader, param_path, class_labels):
    checkpoint = torch.load(param_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    classes = list(class_labels)

    # 收集真实标签和预测结果
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for img_data, id_data, bbox_data in test_loader:
            img_data = img_data.to(device)
            id_data = id_data.to(device)
            outputs = model(img_data)
            
            # 调试输出形状
            print("Output shapes:", outputs.shape)
            
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果
            print(f"predicted: {predicted}, \nid_data: {id_data}")
            true_labels.extend(id_data.cpu().numpy()) # 将 Tensor 转为 NumPy 数组
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
    plot_confusion_matrix(cm, class_labels)

if __name__ == '__main__':
    dataset = FaceDataset(DATASET_ROOT_DIR, input_size=224, train_split_ratio=TRAIN_SPLIT_RATIO)
    train_dataset, test_dataset, class_to_idx = dataset.get_datasets()
    if train_dataset is None or test_dataset is None:
        print("数据集加载失败，请检查路径和文件格式。")
        exit(1)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = FaceIDResNetV1(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=MODEL_TYPE_NUM)
    test(model, test_loader, MODEL_SAVE_PATH, class_to_idx.keys())