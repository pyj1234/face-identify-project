# model_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes=1000):
        """
        构建一个简化的10层人脸识别模型 (PyTorch)。
        这里的层数计算方式与 Keras 示例中的 model_balanced_layers 类似，
        主要计算 Conv2d 和 MaxPool2d 层。

        参数:
            num_classes: 数据集中人脸的类别数。
        """
        super(FaceRecognitionModel, self).__init__()

        # 卷积层 + 激活 + 批归一化 + 池化
        # 第 1 层 Conv + Pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 1

        # 第 2 层 Conv + Pool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 2

        # 第 3 层 Conv + Pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 3

        # 第 4 层 Conv + Pool
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 4

        # 第 5 层 Conv + Pool
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 5 (核心处理层已达10 = 5 Conv + 5 Pool)

        # 全连接层
        # 需要计算经过卷积和池化后特征图的尺寸，这里假设输入是 112x112
        # 112 -> 56 -> 28 -> 14 -> 7 -> 3 (Approx after 5 MaxPool2d with kernel 2, stride 2)
        # 最终特征图尺寸大概是 512 * 3 * 3 (取决于输入尺寸和padding/stride)
        # 为了通用性，可以在 forward 中动态计算 Flatten 后的输入尺寸
        self.fc1_input_features = 512 * 3 * 3 # 示例值，请根据实际输入和网络结构调整
        self.fc1 = nn.Linear(self.fc1_input_features, 512) # 6 (全连接层作为主要处理层之一)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

        # 输出层
        self.fc2 = nn.Linear(512, num_classes) # 7 (输出分类层，也是主要处理层)

        # 这里的“10层网络”是一个相对宽松的说法，通常指连续的主要处理模块数量。
        # 这个模型包含 5组 (Conv+BN+ReLU+Pool) 结构，以及 Flatten, FC+BN+ReLU+Dropout, FC，
        # 如果只算 Conv 和 Pool，有10层。如果算 Conv, Pool, FC，有 5+5+2=12层。
        # 如果严格只计算Conv和Dense(Linear)层，那只有 5+2=7层。
        # PyTorch 的 Module 结构更容易灵活组合，不像 Sequential 那样严格线形。
        # 我将使用包含5组Conv+Pool和2个FC的结构，这通常被认为是10+层的网络深度。


    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Conv Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # Conv Block 5
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        # Flatten
        # 动态计算 Flatten 后的尺寸
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # 更新 fc1_input_features 如果需要
        # self.fc1 = nn.Linear(x.size(1), 512) # 不应该在 forward 中重新定义层

        # 如果上面动态计算了尺寸，这里应该使用该尺寸
        # x = x.view(batch_size, self.fc1_input_features) # 如果提前计算好 self.fc1_input_features

        # 为了确保 FC 层尺寸匹配，可以在第一次 forward 时计算
        if not hasattr(self, '_is_flattened'):
             # 在第一次 forward 时计算展平后的特征数量
             num_features = x.size(1)
             self.fc1 = nn.Linear(num_features, 512).to(x.device) # 确保在同一设备
             self.bn6 = nn.BatchNorm1d(512).to(x.device) # 确保在同一设备
             # 不需要在这里重新定义 self.fc2，它已经在 __init__ 中定义了
             print(f"Flattened feature size dynamically calculated: {num_features}")
             self._is_flattened = True # 标记已计算过

        # 全连接层
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)

        # 输出层
        x = self.fc2(x) # Logits

        return x

if __name__ == '__main__':
    # 示例用法
    num_classes_example = 1000
    model = FaceRecognitionModel(num_classes=num_classes_example)
    # 打印模型结构
    print(model)

    # 打印模型参数总量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # 示例输入 (Batch size = 1, Channels=3, Height=112, Width=112)
    input_tensor = torch.randn(1, 3, 112, 112)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}") # 应该是 [Batch size, num_classes]

    # 如果在 forward 中动态计算了 fc1，这里需要实际运行一次 forward 才能看到正确的 FC 层尺寸
    # 可以通过创建一个 dummy input 来触发
    # model = FaceRecognitionModel(num_classes=num_classes_example) # 重新创建模型
    # dummy_input = torch.randn(1, 3, 112, 112)
    # _ = model(dummy_input)
    # print("Model structure after dynamic sizing:")
    # print(model)