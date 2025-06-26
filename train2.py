import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from ann_layers import ANN_Blocks
from Res2Net_v1b import res2net50_v1b_26w_4s
from model2 import FaceIDResNet
from dataset_utils1 import FaceDataset

# 数据集根目录
DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
MODEL_DIR = 'param' # 模型参数保存目录
MODEL_SAVE_PATH = MODEL_DIR + '/face_recognition_onet.pth' # 模型保存路径

# 模型和训练参数
IMG_SIZE = 224 # 输入图像尺寸
BATCH_SIZE = 32
EPOCHS = 50  # 训练轮数
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
        self.conf_loss_fn = nn.BCEWithLogitsLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()
        self.id_loss_fn = nn.CrossEntropyLoss()

        # 初始化优化器和学习率调度器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)

    def assign_targets(self, bbox_data, grid_size=7):
        batch_size = bbox_data.size(0)
        conf_target = torch.zeros(batch_size, 1, grid_size, grid_size).to(bbox_data.device)
        bbox_target = torch.zeros(batch_size, 4, grid_size, grid_size).to(bbox_data.device)
        for b in range(batch_size):
            x, y, w, h = bbox_data[b]
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            conf_target[b, 0, grid_y, grid_x] = 1
            bbox_target[b, :, grid_y, grid_x] = bbox_data[b]
        return conf_target, bbox_target
    
    def train(self):
        best_val_accuracy = 0.0
        if os.path.exists(self.param_path):
            try:
                checkpoint = torch.load(self.param_path, map_location=self.device)
                model_state_dict = checkpoint['model_state_dict']
                old_best_val_accuracy = checkpoint['best_val_accuracy']
                class_mapping = checkpoint['class_to_idx']

                best_val_accuracy = old_best_val_accuracy
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
            total_conf, total_bbox, total_id = 0, 0, 0

            for i, (img_data, id_data, bbox_data) in enumerate(self.train_loader):
                img_data, id_data, bbox_data = img_data.to(self.device), id_data.to(self.device), bbox_data.to(self.device) / 224.0
                
                outputs = self.model(img_data)
                # print("\n当前预测结果：" + str(outputs))
                # print("\n当前预测结果identity_pred：" + str(outputs['identity_pred']))

                # confidence = outputs['confidence'].mean(dim=(2, 3))
                # bbox_pred = outputs['bbox'].mean(dim=(2, 3))
                confidence = outputs['confidence']
                bbox_pred = outputs['bbox']
                id_pred = outputs['identity_pred']
                # conf_target = torch.ones_like(confidence)

                # 分配目标
                conf_target, bbox_target = self.assign_targets(bbox_data, grid_size=7)

                conf_loss = self.conf_loss_fn(confidence, conf_target)
                mask = (conf_target > 0).expand_as(bbox_pred) 
                bbox_loss = self.bbox_loss_fn(bbox_pred[mask], bbox_target[mask])
                id_loss = self.id_loss_fn(id_pred, id_data)

                print(f"当前损失 - Conf: {conf_loss}, BBox: {bbox_loss}, ID: {id_loss}")
                loss = 1.0 * conf_loss + 1.0 * bbox_loss + 5.0 * id_loss

                # 梯度清零、反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                 # 记录损失
                total_loss += loss.item()
                total_conf += conf_loss.item()
                total_bbox += bbox_loss.item()
                total_id += id_loss.item()

                # 计算训练准确率
                _, predicted = torch.max(id_pred, 1)
                total += id_data.size(0)

                print("预测结果:", predicted)
                print("真实标签:", id_data)
                correct += (predicted == id_data).sum().item()

            train_accuracy = correct / total
            total_loss /= len(self.train_loader)
            total_conf /= len(self.train_loader)
            total_bbox /= len(self.train_loader)
            total_id /= len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}, "
                  f"Conf_loss: {total_conf:.4f}, Bbox_loss: {total_bbox:.4f}, ID_loss: {total_id:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}")

            # 更新学习率
            self.scheduler.step()

            if train_accuracy >= best_val_accuracy:
                best_val_accuracy = train_accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'class_to_idx': self.class_to_idx,
                    # 'optimizer_state_dict': optimizer.state_dict(), # 如果需要保存优化器状态以便继续训练
                }, self.param_path)
                print(f"保存了训练集准确率最高的模型状态和类别映射到: {self.param_path}")
        print("训练完成！")

if __name__ == '__main__':
    # Dataset and DataLoader
    dataset = FaceDataset(DATASET_ROOT_DIR, input_size=224, train_split_ratio=TRAIN_SPLIT_RATIO)
    train_dataset, test_dataset, class_to_idx = dataset.get_datasets()
    if train_dataset is None or test_dataset is None:
        print("数据集加载失败，请检查路径和文件格式。")
        exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model = FaceIDResNet(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=MODEL_TYPE_NUM)
    trainer = Trainer(model, MODEL_SAVE_PATH, train_loader, class_to_idx=class_to_idx, batch_size=BATCH_SIZE, lr=1e-3, epochs=EPOCHS)
    trainer.train()