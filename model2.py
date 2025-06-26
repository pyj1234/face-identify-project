import torch
import torch.nn as nn
import ann_layers
from Res2Net_v1b import res2net50_v1b_26w_4s
from insightface.app import FaceAnalysis

# 专注人脸识别的ResNet模型
class FaceIDResNetV1(nn.Module):
    def __init__(self, res2net, num_classes=40, input_size=224):
        super(FaceIDResNetV1, self).__init__()
        self.res2net = res2net
        # 冻结前几层骨干网络
        for name, param in self.res2net.named_parameters():
            if 'layer4' not in name:  # 只训练最后阶段
                param.requires_grad = False
                
        # 增强的身份识别头
        self.identity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # 每个特征图片的空间维度被池化到 1x1
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.res2net(x)[-1]  # [batch_size, 2048, 7, 7]
        identity = self.identity_head(features)
        return identity  # 直接输出分类结果

# 专注人脸识别的ResNet模型，包含检测头和身份识别头
class FaceIDResNetV2(nn.Module):
    def __init__(self, resnet, num_classes=40, input_size=224):
        super(FaceIDResNetV2, self).__init__()
        self.resnet = resnet
        # 解冻所有层
        for param in self.resnet.parameters():
            param.requires_grad = True
            
        # 共享特征层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1)  # 4 for bbox + 1 for confidence
        )
        
        # 身份识别头（使用预训练权重）
        self.identity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 加载预训练的身份识别权重
        pretrained_dict = torch.load('param/best_identity_model.pth')['state_dict']
        model_dict = self.state_dict()
        # 筛选出identity_head的权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'identity_head' in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        features = self.resnet(x)[-1]
        shared = self.shared_conv(features)
        
        # 检测输出
        detection = self.detection_head(shared)
        confidence = torch.sigmoid(detection[:,:1])  # [batch_size, 1, 7, 7]
        bbox = detection[:,1:]  # [batch_size, 4, 7, 7]
        
        # 身份输出
        identity = self.identity_head(shared)
        
        return {'confidence': confidence, 'bbox': bbox, 'identity_pred': identity}

class FaceIDResNet(nn.Module):
    def __init__(self, resnet, num_classes=40, input_size=224):
        super(FaceIDResNet, self).__init__()
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1),  # Confidence
            nn.Conv2d(2048, 4, kernel_size=1)   # Bounding box
        )
        self.identity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [batch_size, 2048, 1, 1]
            nn.Flatten(),
            ann_layers.ANN_Blocks(in_features=2048, dropout_val=0.3, out_features=16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)[-1]  # [batch_size, 2048, 7, 7]
        confidence = torch.sigmoid(self.detection_head[0](features))  # [batch_size, 1, 7, 7]
        bbox = self.detection_head[1](features)  # [batch_size, 4, 7, 7]
        identity = self.identity_head(features)  # [batch_size, num_classes]
        return {'confidence': confidence, 'bbox': bbox, 'identity_pred': identity}

if __name__ == '__main__':
    num_classes_example = 40
    model = FaceIDResNetV1(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=num_classes_example)
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output:{str(output)}")
    # print(f"Confidence shape: {output['confidence'].shape}")
    # print(f"BBox shape: {output['bbox'].shape}")
    # print(f"Identity prediction shape: {output['identity_pred'].shape}")

    # print("--------------------------------------------------------")
    # print(f"Confidence: {output['confidence']}")
    # print(f"BBox: {output['bbox']}")
    # print(f"Identity prediction: {output['identity_pred']}")