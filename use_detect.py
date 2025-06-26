# face_recognition_app.py
import sys
import os
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms
from PIL import Image
import numpy as np

# 导入之前定义的模型和工具
from model import FaceRecognitionModel
# from dataset_utils import load_and_split_dataset # 在这里不需要加载整个数据集，只需要class_to_idx
# 我们可以直接从保存的模型文件中加载 class_to_idx

# 模型路径 (请根据您的实际情况修改)
MODEL_PATH = 'face_recognition_model_pytorch.pth'

# 模型参数
IMG_HEIGHT, IMG_WIDTH = 112, 112

class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch 人脸识别应用")
        self.setGeometry(100, 100, 600, 700) # 设置窗口大小

        self.image_path = None
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None # 索引到类别的映射

        self.initUI()
        self.load_model() # 在应用启动时加载模型

    def initUI(self):
        # 主布局
        main_layout = QVBoxLayout()

        # 图片显示区域
        self.image_label = QLabel("请选择一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        # 设置 QLabel 尺寸策略，使其可以扩展并保持内容居中
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False) # 不拉伸图片，保持比例
        self.image_label.setStyleSheet("border: 1px solid grey;") # 添加边框以便区分

        # 按钮布局
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("选择图片")
        self.recognize_button = QPushButton("识别")
        # 初始时识别按钮可能不可用，直到图片被加载
        self.recognize_button.setEnabled(False)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.recognize_button)

        # 结果显示区域
        self.result_label = QLabel("识别结果:")
        self.result_label.setAlignment(Qt.AlignLeft)

        # 添加组件到主布局
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

        # 连接信号和槽
        self.select_button.clicked.connect(self.select_image)
        self.recognize_button.clicked.connect(self.recognize_face)

    def load_model(self):
        """
        加载训练好的 PyTorch 模型和类别映射。
        """
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型加载设备: {device}")

        if not os.path.exists(MODEL_PATH):
            self.result_label.setText(f"错误: 未找到模型文件 {MODEL_PATH}。")
            print(f"错误: 未找到模型文件 {MODEL_PATH}。")
            return

        try:
            # 加载模型状态字典和类别映射
            print(f"正在加载模型状态字典和类别映射从: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model_state_dict = checkpoint['model_state_dict']
            self.class_to_idx = checkpoint['class_to_idx']
            num_classes = len(self.class_to_idx)
            print(f"模型状态和类别映射加载成功，类别数: {num_classes}")

            # 创建索引到类别的映射
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            # 构建模型实例并加载状态字典
            self.model = FaceRecognitionModel(num_classes=num_classes)
            self.model.load_state_dict(model_state_dict)
            self.model.to(device) # 确保模型在正确设备上
            self.model.eval() # 设置模型为评估模式
            print("模型加载成功，可以开始识别。")

        except Exception as e:
            self.result_label.setText(f"错误: 加载模型失败 - {e}")
            print(f"错误: 加载模型失败 - {e}")
            self.model = None # 加载失败，模型设置为 None

    def select_image(self):
        """
        打开文件对话框选择图片文件。
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # 可选：不使用原生对话框
        # PGM 文件过滤器，可以添加其他图片格式如 *.jpg *.png
        file_filter = "图片文件 (*.pgm *.jpg *.jpeg *.png);;PGM 文件 (*.pgm);;所有文件 (*)"
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "", # 默认目录
            file_filter,
            options=options
        )
        if fileName:
            self.image_path = fileName
            print(f"选择了文件: {self.image_path}")
            self.display_image(self.image_path)
            # 图片加载成功后启用识别按钮
            self.recognize_button.setEnabled(True)
            self.result_label.setText("图片已加载，请点击识别按钮。")

    def display_image(self, path):
        """
        在 QLabel 中显示图片。
        """
        if path:
            try:
                # 使用 Pillow 打开图片，转换为 RGB 确保兼容性
                img = Image.open(path).convert("RGB")

                # 将 PIL Image 转换为 QImage
                # PIL Image mode 'RGB' is 24-bit RGB
                # QImage format must match
                if img.mode == 'RGB':
                     # Image.tobytes() gives data row by row
                    qimage = QImage(img.tobytes(), img.size[0], img.size[1], img.size[0] * 3, QImage.Format_RGB888)
                elif img.mode == 'L': # PGM 原始模式可能是灰度 L
                     qimage = QImage(img.tobytes(), img.size[0], img.size[1], img.size[0] * 1, QImage.Format_Grayscale8)
                else:
                     # 尝试转换为 RGB 再处理，或者处理其他格式
                     img = img.convert('RGB')
                     qimage = QImage(img.tobytes(), img.size[0], img.size[1], img.size[0] * 3, QImage.Format_RGB888)


                # 将 QImage 转换为 QPixmap，并缩放以适应 QLabel
                pixmap = QPixmap.fromImage(qimage)

                # 根据 QLabel 的当前尺寸按比例缩放图片
                # self.image_label.size() 获取当前 QLabel 的尺寸
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio, # 保持宽高比
                    Qt.SmoothTransformation # 平滑缩放
                ))
                self.image_label.setText("") # 清除“请选择图片”文本

            except Exception as e:
                self.image_label.setText(f"加载图片失败: {e}")
                print(f"加载图片失败: {e}")
                self.recognize_button.setEnabled(False) # 加载失败，禁用识别按钮


    def preprocess_image(self, image_path):
        """
        对图片进行预处理，使其符合模型输入要求。
        """
        try:
            # 使用 Pillow 打开图片，转换为 RGB 确保3通道输入
            img = Image.open(image_path).convert("RGB")

            # 定义与训练时一致的转换 (没有数据增强)
            preprocess = transforms.Compose([
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # 缩放到模型输入尺寸
                transforms.ToTensor(), # 转换为 Tensor ([C, H, W]), 像素值 [0, 1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
            ])

            # 应用转换，并在批次维度上扩展 (因为模型期望批次输入)
            img_tensor = preprocess(img).unsqueeze(0) # 添加批次维度 [1, C, H, W]

            # 将 tensor 移动到模型所在的设备
            device = next(self.model.parameters()).device # 获取模型当前所在的设备
            img_tensor = img_tensor.to(device)

            return img_tensor

        except Exception as e:
            self.result_label.setText(f"图片预处理失败: {e}")
            print(f"图片预处理失败: {e}")
            return None

    def recognize_face(self):
        """
        使用加载的模型识别当前显示的图片中的人脸。
        """
        if self.model is None:
            self.result_label.setText("模型未加载或加载失败。")
            print("模型未加载或加载失败，无法识别。")
            return

        if self.image_path is None:
            self.result_label.setText("请先选择一张图片。")
            print("未选择图片，无法识别。")
            return

        # 预处理图片
        input_tensor = self.preprocess_image(self.image_path)
        if input_tensor is None:
            return # 预处理失败，错误信息已在 preprocess_image 中显示

        print("开始进行人脸识别...")
        try:
            # 在 no_grad 模式下进行推理
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # 获取预测概率
            probabilities = F.softmax(outputs, dim=1)

            # 获取最高概率的类别索引和概率值
            # torch.max(probabilities, 1) 返回 (max_values, max_indices)
            max_prob, predicted_idx_tensor = torch.max(probabilities, 1)
            predicted_idx = predicted_idx_tensor.item() # 从 tensor 获取 Python int
            confidence = max_prob.item() # 从 tensor 获取 Python float

            # 将预测索引映射回类别名称 (例如 's1', 's2')
            predicted_class_name = self.idx_to_class.get(predicted_idx, "未知类别")

            # 显示结果
            result_text = f"识别结果: {predicted_class_name} (置信度: {confidence:.4f})"
            self.result_label.setText(result_text)
            print(result_text)

        except Exception as e:
            self.result_label.setText(f"识别过程中发生错误: {e}")
            print(f"识别过程中发生错误: {e}")


# 应用主入口
if __name__ == '__main__':
    # 在 Windows 下，如果使用多进程 (例如 DataLoader num_workers > 0),
    # 并且在主程序中包含了要被子进程 import 的部分（比如这里的模型和数据集类），
    # 最好加上 multiprocessing.freeze_support() 或放在 if __name__ == '__main__': 保护块内。
    # 对于这个简单的应用，模型加载在主进程，DataLoader 不在应用代码中直接使用，所以通常不是问题，
    # 但这是一个好的实践。
    # import multiprocessing
    # multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    ex = FaceRecognitionApp()
    ex.show()
    sys.exit(app.exec_())