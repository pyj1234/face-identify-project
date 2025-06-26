import torch
from torch import nn
import torch.nn.functional as F

class FaceRecognitionONet(nn.Module):
    def __init__(self, num_classes=40, input_size=48):
        """
        基于 ONet 特征提取部分构建的人脸识别模型。
        参数:
            num_classes (int): 数据集中人脸的类别数
        """
        super(FaceRecognitionONet, self).__init__()
        self.input_size = input_size  # 默认输入尺寸为48x48

        # 从 ONet 复制特征提取层
        self.feature_extractor = nn.Sequential(
            # 第1层卷积
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1), # 46x46
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(num_parameters=32),

            # 第1层池化
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 23x23

            # 第2层卷积
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0), # 21x21
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),

            # 第2层池化
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 10x10

            # 第3层卷积
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0), # 8x8
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),

            # 第3个池化层
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 4x4

            # 第4层卷积
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0), # 3x3
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(num_parameters=128)
        )

        # 从 ONet 复制第一个全连接层，降维至256维
        self.linear5 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU(num_parameters=256)
        )

        # 添加新的分类头，用于区分 40 个人脸类别
        self.classification_head = nn.Linear(in_features=256, out_features=num_classes)

        # 初始化 PReLU 层的参数 (如果您想使用 ONet 代码中的 init=0.25)
        for m in self.modules():
            if isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.25)


    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x) # 期望输入 48x48 图片

        # 展平
        #x = x.view(-1, 128 * 3 * 3) # 也可以使用 x.view(x.size(0), -1) 更灵活
        x = torch.flatten(x, 1) # 使用 torch.flatten 从 dim 1 开始展平

        # 通过第一个全连接层
        x = self.linear5(x)

        # 通过分类头获取分类 logits
        logits = self.classification_head(x)

        return logits # 对于 CrossEntropyLoss，直接返回 logits

if __name__ == '__main__':
    num_classes_example = 40
    model = FaceRecognitionONet(num_classes=num_classes_example)
    # 打印模型结构
    print(model)

    # 打印模型参数总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # 示例输入 (Batch size = 1, Channels=3, Height=48, Width=48)
    # 注意：ONet 期望的输入是 48x48，所以我们的数据加载和预处理也需要调整到 48x48
    input_tensor = torch.randn(1, 3, 48, 48)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}") # 应该是 [Batch size, num_classes]