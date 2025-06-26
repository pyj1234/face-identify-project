import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from face_detect import extract_highest_confidence_face_resized
import glob
import cv2
import os

import cv2
import numpy as np
import random

class IDataSet(Dataset):
    def __init__(self, data_list, class_to_idx, input_size=224):
        self.color_jitter = (0.2, 0.2, 0.2)  # 亮度、对比度、饱和度
        self.grayscale_prob = 0.1
        self.erase_prob = 0.3
        self.erase_scale = (0.02, 0.1)

        self.data_list = data_list
        self.class_to_idx = class_to_idx
        self.input_size = input_size

        self.images = []    # 存储图片路径
        self.labels = []    # 存储标签
        self.bboxes = []    # 存储边界框
        for (img_path, label, bbox) in data_list:
            self.images.append(img_path)
            self.labels.append(label)
            self.bboxes.append(bbox)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        extracted_face = extract_highest_confidence_face_resized(
            self.images[idx],
        )
        # img = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (self.input_size, self.input_size))
        img = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB) 

        # 数据增强
        if random.random() < 0.5:
            img = self._horizontal_flip(img)
        if random.random() < 0.5:
            img = self._color_jitter(img)
        if random.random() < self.grayscale_prob:
            img = self._random_grayscale(img)
        if random.random() < self.erase_prob:
            img = self._random_erasing(img)

        # 转换为Tensor并归一化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 手动标准化（使用ImageNet均值和标准差）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        bbox = torch.tensor(self.bboxes[idx], dtype=torch.float)

        return img, label, bbox
    # 水平翻转
    def _horizontal_flip(self, img):
        return cv2.flip(img, 1)  # 1表示水平翻转
    # 色彩抖动
    def _color_jitter(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 亮度调整
        v = np.clip(v * random.uniform(1 - self.color_jitter[0], 1 + self.color_jitter[0]), 0, 255).astype(np.uint8)
        # 饱和度调整
        s = np.clip(s * random.uniform(1 - self.color_jitter[1], 1 + self.color_jitter[1]), 0, 255).astype(np.uint8)
        # 色相调整
        h = np.clip((h + random.uniform(-self.color_jitter[2], self.color_jitter[2]) * 180) % 180, 0, 179).astype(np.uint8)
        
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 随机灰度化
    def _random_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 随机遮挡
    def _random_erasing(self, img):
        h, w, _ = img.shape
        # 随机生成遮挡区域
        erase_area = np.prod([h, w]) * random.uniform(self.erase_scale[0], self.erase_scale[1])
        aspect_ratio = random.uniform(0.3, 3.3)
        
        erase_h = int(np.sqrt(erase_area / aspect_ratio))
        erase_w = int(np.sqrt(erase_area * aspect_ratio))
        
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        
        # 用随机噪声填充遮挡区域
        img[y:y+erase_h, x:x+erase_w] = np.random.randint(
            0, 256, (erase_h, erase_w, 3), dtype=np.uint8
        )
        return img

class FaceDataset():
    def __init__(self, data_path, input_size=224, train_split_ratio=0.3):
        self.data_path = data_path
        self.input_size = input_size
        self.train_split_ratio = train_split_ratio  # 训练集比例

        self.classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.startswith('s')],
                                key=lambda x: int(x[1:])) # 按 s 后面的数字排序
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        if not self.classes:
            print(f"错误: 在目录 {data_path} 中未找到以 's' 开头且包含数字的子文件夹。")
            return None, None, None
        print(f"检测到 {len(self.classes)} 个类别: {self.classes}")

        self.train_subset_list = []
        self.test_subset_list = []
        for cls in self.classes:
            cls_path = os.path.join(data_path, cls)
            pgm_files = glob.glob(os.path.join(cls_path, '*.pgm'))

            if not pgm_files:
                print(f"警告: 类别 {cls} ({cls_path}) 下没有找到 .pgm 图片文件，该类别将被跳过。")
                continue # 跳过没有图片的类别

            full_data_list = [] # 用于存储当前类别的所有图片路径和标签

            label = self.class_to_idx[cls]
            cur_class_pgm_size = len(pgm_files)
            train_size = int(cur_class_pgm_size * self.train_split_ratio)
            test_size = cur_class_pgm_size - train_size
            for img_path in pgm_files:
                box = [0, 0, self.input_size, self.input_size]
                full_data_list.append((img_path, label, box)) # 直接存储路径、标签和边界框

            # 构建完整的数据列表 (路径和标签)
            if len(full_data_list) == 0:
                print("错误: 数据列表为空。")
                return None, None, None
            
            # 这里 random_split 返回的是 Subset 对象
            train_subset, test_subset = random_split(full_data_list, [train_size, test_size])
            self.train_subset_list.extend(train_subset)
            self.test_subset_list.extend(test_subset)

        if not self.train_subset_list or not self.test_subset_list:
            print(f"错误: 在目录 {self.data_path} 的子文件夹中未找到任何 .pgm 图片文件。")
            return None, None, None
        print(f"总共找到 {len(self.train_subset_list) + len(self.test_subset_list)} 张图片。")

        self.train_dataset = IDataSet(data_list=self.train_subset_list, class_to_idx=self.class_to_idx)
        self.test_dataset = IDataSet(data_list=self.test_subset_list, class_to_idx=self.class_to_idx)

    def get_datasets(self):
        return self.train_dataset, self.test_dataset, self.class_to_idx