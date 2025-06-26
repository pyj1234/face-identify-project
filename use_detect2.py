# face_recognition_app_pyqt.py
import sys
import os
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSizePolicy,
                             QComboBox, QSpacerItem, QMessageBox) # 导入 QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QRect, QThread, pyqtSignal, pyqtSlot, QSize
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 # 导入 OpenCV

# 导入之前定义的模型和工具
from model2 import FaceIDResNetV1 # Ensure model2.py is in the same directory or PYTHONPATH
from Res2Net_v1b import res2net50_v1b_26w_4s # Ensure Res2Net_v1b.py is accessible
from use_face_detect import extract_highest_confidence_face_resized

# 人脸识别模型路径
MODEL_PATH = 'param/face_recognition_onet.pth'

# # Haar Cascade 人脸检测器路径
# HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# DNN 人脸检测器模型路径
DNN_PROTO_PATH = './deploy.prototxt' # 确保此路径存在
DNN_MODEL_PATH = './res10_300x300_ssd_iter_140000.caffemodel'

# 模型参数
IMG_SIZE = 224 # 输入图像尺寸 (用于模型)
MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH = IMG_SIZE, IMG_SIZE

MODEL_TYPE_NUM = 40 # 模型类别数

# 数据标准化参数
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


class VideoThread(QThread):
    """
    用于在单独线程中捕获视频帧并处理。
    """
    # 定义信号，用于将处理后的帧发送回主线程
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str) # 新增错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        
        self.model = None
        self.idx_to_class = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 预处理转换
        self.preprocess = transforms.Compose([
            # transforms.ToPILImage() # 确保输入是 PIL Image # Input will be PIL Image
            transforms.Resize((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)), # 缩放到模型输入尺寸
            transforms.ToTensor(), # 转换为 Tensor ([C, H, W]), 像素值 [0, 1]
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD) # 标准化
        ])

    def run(self):
        if not self._run_flag: # 如果 Haar Cascade 未加载成功，直接退出
            return

        # 捕获视频
        camera_index = 0  # 默认摄像头索引
        if isinstance(self.parent(), FaceRecognitionApp):
            camera_index = self.parent().camera_dropdown.currentData()
            if camera_index is None or not isinstance(camera_index, int):
                self.error_signal.emit("错误: 视频线程未获取到有效的摄像头索引。")
                self._run_flag = False
                return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.error_signal.emit(f"错误: 无法打开摄像头 {camera_index}。请检查摄像头是否连接或是否被其他应用占用。")
            self._run_flag = False
            return

        # 获取视频帧的原始尺寸
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头 {camera_index} 开启，分辨率: {frame_width}x{frame_height}")

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # 人脸检测
                images_list, faces_list = extract_highest_confidence_face_resized(cv_img)

                # 在原始帧上绘制矩形和标签，并进行识别
                if self.model is not None and self.idx_to_class is not None:
                    for i, (x, y, w, h) in enumerate(faces_list):
                        face_img_cv = images_list[i] if i < len(images_list) else None
                        if face_img_cv is None:
                            continue
                        if face_img_cv.size > 0: # 检查裁剪区域是否有效
                            try:
                                face_img_rgb = cv2.cvtColor(face_img_cv, cv2.COLOR_BGR2RGB)
                                
                                # 将 OpenCV RGB 图片 (numpy array) 转为 PIL Image，以便进行 PyTorch 预处理
                                face_img_pil = Image.fromarray(face_img_rgb)

                                # 预处理
                                input_tensor = self.preprocess(face_img_pil).unsqueeze(0).to(self.device)

                                # 识别 (推理)
                                with torch.no_grad():
                                    outputs = self.model(input_tensor)
                                    # 将 logits 转换为概率
                                    probabilities = F.softmax(outputs, dim=1) 
                                    max_prob, predicted_idx_tensor = torch.max(probabilities, 1)
                                    
                                    predicted_idx = predicted_idx_tensor.item()
                                    confidence = max_prob.item()

                                predicted_class_name = self.idx_to_class.get(predicted_idx, "未知")

                                # 在帧上绘制矩形和标签
                                color = (0, 255, 0) # BGR 格式，这里是绿色
                                cv2.rectangle(cv_img, (x, y), (x+w, y+h), color, 2)

                                # 标签文本
                                label_text = f"{predicted_class_name}: {confidence:.2f}"
                                # 获取文本大小以便确定放置位置
                                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                # 放置标签文本 (稍微上移一点)
                                cv2.rectangle(cv_img, (x, y - text_height - baseline), (x + text_width, y), color, -1) # 绘制背景框
                                cv2.putText(cv_img, label_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # 绘制文本

                            except Exception as e:
                                print(f"视频流识别或绘制错误: {e}")
                                # 即使出错，也绘制检测框，并标记为“识别失败”
                                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # 红色框表示出错
                                error_text = "识别失败"
                                (text_width, text_height), baseline = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                cv2.rectangle(cv_img, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 255), -1)
                                cv2.putText(cv_img, error_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                                
                # 发送处理后的帧信号
                self.change_pixmap_signal.emit(cv_img)

        # 释放摄像头
        cap.release()
        print("视频线程已停止并释放摄像头。")

    def stop(self):
        """停止线程."""
        self._run_flag = False
        self.wait()

    def set_model(self, model, class_to_idx):
        """设置识别模型和类别映射."""
        self.model = model
        self.model.eval() # 确保模型在评估模式
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        print("识别模型和类别映射已设置到视频线程。")


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt 人脸检测与识别应用")
        self.setGeometry(100, 100, 800, 800) # 设置窗口大小
        self.setMinimumSize(QSize(300, 300))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_path = None
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None

        self.video_thread = None # 视频处理线程实例

        self.initUI()
        self.load_model() # 在应用启动时加载模型
        # 初始化摄像头下拉框，加载模型后才能设置好视频线程
        self.init_camera_dropdown() 
        
        # 预处理转换 (与训练时保持一致), input should be PIL Image
        self.preprocess = transforms.Compose([
            transforms.Resize((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)), # 缩放到模型输入尺寸
            transforms.ToTensor(), # 转换为 Tensor ([C, H, W]), 像素值 [0, 1]
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD) # 标准化
        ])

    def initUI(self):
        # 主布局
        main_layout = QVBoxLayout()

        # 图片/视频显示区域
        self.display_label = QLabel("请选择图片或启动视频")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setScaledContents(False) # 不拉伸内容，保持比例
        self.display_label.setStyleSheet("border: 1px solid grey; background-color: black;") # 添加边框和黑色背景

        self.display_label.setMaximumSize(QSize(2000, 1000))

        # 控制按钮和下拉框布局
        control_layout = QHBoxLayout()

        # 图片操作按钮
        self.select_button = QPushButton("选择图片")
        self.recognize_button = QPushButton("识别图片")
        self.recognize_button.setEnabled(False) # 初始禁用识别按钮

        control_layout.addWidget(self.select_button)
        control_layout.addWidget(self.recognize_button)

        # 添加分隔符或空白项
        control_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)) # 可伸缩空白

        # 视频操作控件
        self.camera_dropdown = QComboBox()
        self.camera_dropdown.addItem("选择摄像头...", None) # 默认项，数据为 None
        self.start_video_button = QPushButton("启动视频")
        self.stop_video_button = QPushButton("停止视频")
        self.stop_video_button.setEnabled(False) # 初始禁用停止按钮

        control_layout.addWidget(QLabel("摄像头:"))
        control_layout.addWidget(self.camera_dropdown)
        control_layout.addWidget(self.start_video_button)
        control_layout.addWidget(self.stop_video_button)

        # 结果显示区域 (用于单张图片)
        self.result_label = QLabel("识别结果:")
        self.result_label.setAlignment(Qt.AlignLeft)

        # 添加组件到主布局
        main_layout.addWidget(self.display_label)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.result_label) # 单张图片结果显示

        self.setLayout(main_layout)

        # 连接信号和槽
        self.select_button.clicked.connect(self.select_image)
        self.recognize_button.clicked.connect(self.recognize_static_image)
        self.start_video_button.clicked.connect(self.start_video)
        self.stop_video_button.clicked.connect(self.stop_video)

    def init_camera_dropdown(self):
        """
        检测可用摄像头并填充下拉框。
        """
        print("检测可用摄像头...")
        # 尝试打开摄像头索引 0 到 9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_dropdown.addItem(f"摄像头 {i}", i) # 存储摄像头索引作为数据
                cap.release()
            else:
                # 如果连续几个索引都打不开，可能就没有更多摄像头了
                # 这里的判断可以更智能，例如连续3个打不开就停止
                pass 

        if self.camera_dropdown.count() > 1:
            self.camera_dropdown.setCurrentIndex(1) # 默认选中第一个检测到的摄像头
        else:
            print("未检测到可用摄像头。")
            self.start_video_button.setEnabled(False) # 如果没有摄像头，禁用启动视频按钮

    def load_model(self):
        """
        加载训练好的 PyTorch 模型和类别映射。
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模型加载设备: {device}")

        if not os.path.exists(MODEL_PATH):
            self.result_label.setText(f"错误: 未找到模型文件 {MODEL_PATH}。请确保模型已训练并保存。")
            print(f"错误: 未找到模型文件 {MODEL_PATH}。")
            # QMessageBox.critical(self, "模型加载错误", f"未找到模型文件 {MODEL_PATH}。")
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
            # 确保 MODEL_TYPE_NUM 和 num_classes 匹配
            if num_classes != MODEL_TYPE_NUM:
                print(f"警告: MODEL_TYPE_NUM ({MODEL_TYPE_NUM}) 与加载模型中的类别数 ({num_classes}) 不匹配。将使用加载模型中的类别数。")
                
            self.model = FaceIDResNetV1(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=num_classes) # 使用实际加载的类别数
            self.model.load_state_dict(model_state_dict, strict=False) # strict=False 允许加载部分参数
            self.model.to(device) # 确保模型在正确设备上
            self.model.eval() # 设置模型为评估模式
            print("识别模型加载成功。")
            # QMessageBox.information(self, "模型加载", "识别模型加载成功。")


        except Exception as e:
            self.result_label.setText(f"错误: 加载模型失败 - {e}")
            print(f"错误: 加载模型失败 - {e}")
            QMessageBox.critical(self, "模型加载错误", f"加载模型失败: {e}")
            self.model = None # 加载失败，模型设置为 None

    def select_image(self):
        """
        打开文件对话框选择图片文件。
        """
        # 如果视频正在运行，先停止视频
        if self.video_thread and self.video_thread.isRunning():
            self.stop_video()

        options = QFileDialog.Options()
        file_filter = "图片文件 (*.pgm *.jpg *.jpeg *.png);;PGM 文件 (*.pgm);;JPEG 文件 (*.jpg *.jpeg);;PNG 文件 (*.png);;所有文件 (*)"
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
            self.display_static_image(self.image_path)
            # 图片加载成功且模型加载成功后启用识别按钮
            if self.model is not None:
                self.recognize_button.setEnabled(True)
            self.result_label.setText("图片已加载，请点击识别图片按钮进行检测和识别。")

    def display_static_image(self, path):
        """
        在 display_label 中显示静态图片。
        """
        if path:
            try:
                # 使用 OpenCV 读取图片 (支持更多格式)，默认是 BGR
                cv_img = cv2.imread(path)
                if cv_img is None:
                    raise ValueError("无法读取图片文件，请检查路径和文件格式。")

                # 将 OpenCV BGR 图片转换为 RGB 用于显示，因为 QImage.Format_RGB888 通常更通用
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_img.shape
                bytesPerLine = 3 * width
                qimage = QImage(rgb_img.data, width, height, bytesPerLine, QImage.Format_RGB888) 

                # 将 QImage 转换为 QPixmap，并缩放以适应 QLabel
                pixmap = QPixmap.fromImage(qimage)

                # 根据 QLabel 的当前尺寸按比例缩放图片
                self.display_label.setPixmap(pixmap.scaled(
                    self.display_label.size(),
                    Qt.KeepAspectRatio, # 保持宽高比
                    Qt.SmoothTransformation # 平滑缩放
                ))
                self.display_label.setText("") # 清空文本

            except Exception as e:
                self.display_label.setText(f"加载图片失败: {e}")
                print(f"加载图片失败: {e}")
                self.recognize_button.setEnabled(False) # 加载失败，禁用识别按钮

    # MODIFIED_ участок кода
    def preprocess_image_for_recognition(self, face_img_cv_gray):
        """
        对从 OpenCV 读取的灰度人脸区域图片进行预处理，使其符合识别模型输入要求.
        Input is a 2D grayscale OpenCV image (numpy array).
        Output is a 4D tensor [1, C, H, W] ready for the model.
        """
        try:
            if face_img_cv_gray is None or face_img_cv_gray.size == 0:
                print("无效的人脸图像传入预处理。")
                return None

            # 1. Convert grayscale OpenCV image to RGB OpenCV image, then to PIL Image
            # Model expects 3 channels, NORM_MEAN and NORM_STD are for 3 channels.
            # self.preprocess (transforms.Compose) expects a PIL Image.
            face_img_rgb_cv = cv2.cvtColor(face_img_cv_gray, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(face_img_rgb_cv) # PIL Image in RGB format

            # 2. Apply the standard torchvision transforms (Resize, ToTensor, Normalize)
            # self.preprocess outputs a 3D tensor (C, H, W)
            img_tensor_3d = self.preprocess(pil_img)

            # 3. Add batch dimension
            img_tensor_4d = img_tensor_3d.unsqueeze(0) # Shape: [1, C, H, W]

            # 4. Move to model's device (optional here if model is already on the correct device,
            # but good practice if device might vary or not be known by preprocess)
            if self.model: # Ensure model is loaded
                 device = next(self.model.parameters()).device
                 img_tensor_4d = img_tensor_4d.to(device)

            return img_tensor_4d

        except Exception as e:
            print(f"静态图片人脸预处理失败: {e}")
            # Potentially show error to user via QMessageBox or result_label
            # self.result_label.setText(f"错误: 图片预处理失败 - {e}")
            return None

    def recognize_static_image(self):
        """
        对加载的静态图片进行人脸检测和识别。
        """
        if self.model is None:
            self.result_label.setText("模型未加载。")
            print("模型未加载，无法识别。")
            QMessageBox.warning(self, "识别错误", "模型未加载，无法进行识别。")
            return
        
        if self.image_path is None:
            self.result_label.setText("未选择图片。")
            print("未选择图片，无法识别。")
            QMessageBox.warning(self, "识别错误", "未选择图片，无法进行识别。")
            return

        # 使用 OpenCV 读取原始图片 (灰度格式用于检测)
        cv_img_original = cv2.imread(self.image_path)
        if cv_img_original is None:
            self.result_label.setText("错误: 无法读取选定的图片文件。")
            print("错误: 无法读取选定的图片文件。")
            QMessageBox.critical(self, "文件读取错误", "无法读取选定的图片文件。")
            return

        # 复制一份彩色原图用于绘制结果和显示
        cv_img_display = cv2.imread(self.image_path)
        if cv_img_display is None: # Should not happen if grayscale read worked, but good check
            self.result_label.setText("错误: 无法读取图片用于显示。")
            return

        # 人脸检测
        print("开始进行人脸检测...")
        images_list, faces_list = extract_highest_confidence_face_resized(cv_img_original)

        # 人脸检测
        print(f"检测到 {len(faces_list)} 张人脸。")

        if len(faces_list) == 0:
            self.display_static_image(self.image_path) # 显示原图
            self.result_label.setText("未检测到人脸。")
            return

        self.result_label.setText(f"检测到 {len(faces_list)} 张人脸。正在识别...")
        print("开始进行人脸识别...")
        recognized_faces_info = [] # 存储每张人脸的识别结果

        try:
            if self.model is not None and self.idx_to_class is not None:
                for i, (x, y, w, h) in enumerate(faces_list):
                    face_img_cv = images_list[i] if i < len(images_list) else None
                    if face_img_cv is None:
                        continue

                    if face_img_cv.size > 0: # 检查裁剪区域是否有效
                        try:
                            face_img_rgb = cv2.cvtColor(face_img_cv, cv2.COLOR_BGR2RGB)
                            
                            # 将 OpenCV RGB 图片 (numpy array) 转为 PIL Image，以便进行 PyTorch 预处理
                            face_img_pil = Image.fromarray(face_img_rgb)

                            # 预处理
                            input_tensor = self.preprocess(face_img_pil).unsqueeze(0).to(self.device)

                            # 识别
                            with torch.no_grad():
                                outputs = self.model(input_tensor)
                                # 将 logits 转换为概率
                                probabilities = F.softmax(outputs, dim=1) 
                                max_prob, predicted_idx_tensor = torch.max(probabilities, 1)
                                
                                predicted_idx = predicted_idx_tensor.item()
                                confidence = max_prob.item()

                                print(f"人脸 {i+1} 识别结果: 类别索引 {predicted_idx}, 置信度 {confidence:.2f}")

                            predicted_class_name = self.idx_to_class.get(predicted_idx, "未知")

                            # 在帧上绘制矩形和标签
                            color = (0, 255, 0) # BGR 格式，这里是绿色
                            cv2.rectangle(cv_img_original, (x, y), (x+w, y+h), color, 2)

                            # 标签文本
                            label_text = f"{predicted_class_name}: {confidence:.2f}"
                            # 获取文本大小以便确定放置位置
                            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            # 放置标签文本
                            cv2.rectangle(cv_img_display, (x, y - text_height - baseline), (x + text_width, y), color, -1) # 绘制背景框
                            cv2.putText(cv_img_display, label_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # 绘制文本
                        except Exception as e:
                            print(f"识别或绘制错误: {e}")
                            # 即使出错，也绘制检测框，并标记为“识别失败”
                            cv2.rectangle(cv_img_display, (x, y), (x+w, y+h), (0, 0, 255), 2) # 红色框表示出错
                            error_text = "识别失败"
                            (text_width, text_height), baseline = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(cv_img_display, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 255), -1)
                            cv2.putText(cv_img_display, error_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                        # 在复制的图片上绘制矩形和标签
                        color = (0, 255, 0) # BGR 格式，这里是绿色
                        cv2.rectangle(cv_img_display, (x, y), (x+w, y+h), color, 2)
                    else:
                        recognized_faces_info.append(f"人脸 {i+1}: 无效裁剪区域")


            # 将绘制好矩形和标签的 OpenCV 图片转换为 QPixmap 显示
            rgb_img_display = cv2.cvtColor(cv_img_display, cv2.COLOR_BGR2RGB) # 再次转换为 RGB
            height, width, channel = rgb_img_display.shape
            bytesPerLine = 3 * width
            qimage = QImage(rgb_img_display.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            self.display_label.setPixmap(pixmap.scaled(
                self.display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.display_label.setText("")
            self.result_label.setText("\n".join(recognized_faces_info)) # 显示所有识别结果

        except Exception as e:
            detailed_error = f"识别过程中发生错误: {type(e).__name__} - {e}"
            self.result_label.setText(detailed_error)
            print(detailed_error)
            import traceback
            traceback.print_exc() # Print full traceback to console for debugging
            QMessageBox.critical(self, "识别错误", detailed_error)


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """
        槽函数，用于更新显示区域的视频帧。
        """
        try:
            # 将 OpenCV BGR 图像转换为 RGB 格式，然后创建 QImage
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            
            # 缩放图像以适应 QLabel
            self.display_label.setPixmap(pixmap.scaled(
                self.display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.display_label.setText("") # 清空文本
        except Exception as e:
            print(f"Error updating image in GUI: {e}")
            self.display_label.setText(f"帧更新错误: {e}")


    @pyqtSlot(str)
    def handle_video_thread_error(self, message):
        """
        处理视频线程发出的错误信息。
        """
        QMessageBox.critical(self, "视频流错误", message)
        self.stop_video() # 停止视频，防止持续报错

    def start_video(self):
        """
        启动摄像头视频捕获和处理。
        """
        if self.model is None or self.idx_to_class is None:
            QMessageBox.warning(self, "模型未加载", "请等待识别模型加载完成再启动视频。")
            return
        
        if self.video_thread and self.video_thread.isRunning():
            print("视频已经在运行中。")
            return

        self.result_label.setText("视频流已启动，正在检测和识别...")
        print("启动视频线程...")

        self.video_thread = VideoThread(self)
        # 将信号连接到槽函数
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.error_signal.connect(self.handle_video_thread_error) # 连接错误信号
        

        self.video_thread.set_model(self.model, self.class_to_idx) # 将模型传递给视频线程

        self.video_thread.start() # 启动线程

        self.start_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)
        self.select_button.setEnabled(False) # 视频运行时禁用图片选择
        self.recognize_button.setEnabled(False) # 视频运行时禁用图片识别
        self.camera_dropdown.setEnabled(False)


    def stop_video(self):
        """
        停止摄像头视频捕获。
        """
        if self.video_thread:
            print("停止视频线程...")
            self.video_thread.stop()
            self.video_thread = None # 清除线程引用
            self.start_video_button.setEnabled(True)
            self.stop_video_button.setEnabled(False)
            self.select_button.setEnabled(True) # 视频停止后启用图片选择
            self.camera_dropdown.setEnabled(True)
            if self.image_path and self.model: # 如果有图片加载且模型存在，重新启用识别按钮
                self.recognize_button.setEnabled(True)
            self.display_label.setText("视频已停止。请选择图片或重新启动视频。")
            self.result_label.setText("视频已停止。")
        else:
            print("视频未运行。")

    def closeEvent(self, event):
        """
        关闭窗口时停止视频线程。
        """
        self.stop_video()
        event.accept()

if __name__ == "__main__":
    # Ensure model is correct before starting the app
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. The application might not work correctly.")
        # Optionally, show a pre-app dialog or exit

    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())