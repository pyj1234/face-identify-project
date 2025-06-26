import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import matplotlib.pyplot as plt
from ann_layers import ANN_Blocks
from Res2Net_v1b import res2net50_v1b_26w_4s
from model2 import FaceIDResNetV1
from dataset_utils1 import FaceDataset

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据集根目录
DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
MODEL_DIR = 'param' # 模型参数保存目录
MODEL_SAVE_PATH = MODEL_DIR + '/face_recognition_onet.pth' # 模型保存路径

# 模型和训练参数
IMG_SIZE = 224 # 输入图像尺寸
BATCH_SIZE = 32
EPOCHS = 150  # 训练轮数
TRAIN_SPLIT_RATIO = 0.3 # 训练集比例
MODEL_TYPE_NUM = 40 # 模型类别数

class Trainer:
    def __init__(self, model, param_path, train_loader, class_to_idx, batch_size=32, lr=1e-3, epochs=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.param_path = param_path
        self.train_loader = train_loader
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 初始化优化器和学习率调度器
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)

    # 验证函数
    def evaluate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img_data, id_data, _ in val_loader:
                img_data, id_data = img_data.to(self.device), id_data.to(self.device)
                outputs = model(img_data)
                loss = criterion(outputs, id_data)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += id_data.size(0)
                correct += (predicted == id_data).sum().item()
        
        return val_loss/len(val_loader), correct / total

    # 绘制损失函数和准确率曲线
    def plot_metrics(self, train_losses, train_accuracies):
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-o', label='Training Loss')
        plt.title('训练损失函数曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'r-o', label='Training Accuracy')
        plt.title('训练准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")  # 保存图像
        plt.show()

    def train(self):
        train_losses = []  # 记录每个epoch的训练损失
        train_accuracies = []  # 记录每个epoch的训练准确率
        best_val_accuracy = 0.0 # 初始最优验证准确率
        best_val_loss = 0.0 # 初始最小验证损失
        if os.path.exists(self.param_path):
            try:
                checkpoint = torch.load(self.param_path, map_location=self.device)
                model_state_dict = checkpoint['model_state_dict']
                old_best_val_accuracy = checkpoint['best_val_accuracy']
                old_best_val_loss = checkpoint['best_val_loss']
                class_mapping = checkpoint['class_to_idx']

                print(f"old_best_val_accuracy: {old_best_val_accuracy}")
                print(f"old_best_val_loss: {old_best_val_loss}")

                best_val_accuracy = old_best_val_accuracy
                best_val_loss = old_best_val_loss
                num_classes = len(class_mapping)
                print(f"模型加载成功，类别数: {num_classes}")
            except Exception as e:
                print(f"错误: 加载模型或类别映射失败 - {e}")
                return
            self.model.load_state_dict(model_state_dict, strict=False)
            self.model.to(self.device)
            print("Loaded parameters, continuing training...")
        else:
            print("Initializing parameters, starting training...")
        
        print("开始训练模型...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for i, (img_data, id_data, bbox_data) in enumerate(self.train_loader):
                img_data, id_data, bbox_data = img_data.to(self.device), id_data.to(self.device), bbox_data.to(self.device) / 224.0
                
                outputs = self.model(img_data)
                # print("\n当前预测结果：" + str(outputs))
                # print("\n当前预测结果data：" + str(outputs.data))

                # 前向传播
                loss = self.criterion(outputs, id_data)

                # 梯度清零、反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                 # 记录损失
                total_loss += loss.item()

                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                total += id_data.size(0)

                print("预测结果:", predicted)
                print("真实标签:", id_data)
                correct += (predicted == id_data).sum().item()

            train_accuracy = correct / total
            avg_loss = total_loss / len(self.train_loader)

            train_losses.append(avg_loss)  # 记录损失
            train_accuracies.append(train_accuracy)  # 记录准确率

            # 更新学习率
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, "
                  f"Train Accuracy: {100 * train_accuracy:.6f}%")

            if train_accuracy >= best_val_accuracy:
                if train_accuracy == 1 and avg_loss > best_val_loss:
                    continue
                best_val_accuracy = train_accuracy
                best_val_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'best_val_loss': best_val_loss,
                    'class_to_idx': self.class_to_idx,
                    # 'optimizer_state_dict': optimizer.state_dict(),
                }, self.param_path)
                print(f"保存了训练集准确率最高的模型状态和类别映射到: {self.param_path}")

                # print("验证阶段...")
                # val_loss, val_acc = self.evaluate(model, val_loader, criterion)
                # print(f'Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.6f}%')
        print("训练完成！")
        self.plot_metrics(train_losses=train_losses, train_accuracies=train_accuracies) # 绘制曲线


if __name__ == '__main__':
    dataset = FaceDataset(DATASET_ROOT_DIR, input_size=224, train_split_ratio=TRAIN_SPLIT_RATIO)
    train_dataset, test_dataset, class_to_idx = dataset.get_datasets()
    if train_dataset is None or test_dataset is None:
        print("数据集加载失败，请检查路径和文件格式。")
        exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model = FaceIDResNetV1(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=MODEL_TYPE_NUM)
    trainer = Trainer(model, MODEL_SAVE_PATH, train_loader, class_to_idx=class_to_idx, batch_size=BATCH_SIZE, lr=3e-4, epochs=EPOCHS)
    trainer.train()