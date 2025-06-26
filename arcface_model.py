import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcMarginProduct(nn.Module):
    """ArcFace实现"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class FaceLandmarkNet(nn.Module):
    """人脸关键点检测分支"""
    def __init__(self, in_features=512, num_points=68):
        super(FaceLandmarkNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, num_points*2)  # 每个点有x,y坐标
        )
    
    def forward(self, x):
        return self.fc(x)

class ArcFaceModel(nn.Module):
    """改进的人脸识别模型"""
    def __init__(self, num_classes=40, num_points=68):
        super(ArcFaceModel, self).__init__()
        # 主干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 特征层
        self.feature = nn.Linear(512, 512)
        
        # ArcFace分类头
        self.arcface = ArcMarginProduct(512, num_classes)
        
        # 关键点检测分支
        self.landmark = FaceLandmarkNet(512, num_points)
        
    def forward(self, x, label=None):
        # 特征提取
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        
        # 人脸识别输出
        if label is not None:
            output = self.arcface(features, label)
        else:
            output = F.linear(F.normalize(features), F.normalize(self.arcface.weight))
            output *= self.arcface.s
            
        # 关键点检测
        landmarks = self.landmark(features)
        
        return output, landmarks, features
