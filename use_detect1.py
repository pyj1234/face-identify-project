# face_recognition_app_pyqt.py
import sys
import os
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSizePolicy,
                             QComboBox, QSpacerItem) # 导入 QSpacerItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QRect, QThread, pyqtSignal, pyqtSlot, QSize
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 # 导入 OpenCV

# 导入之前定义的模型和工具
from model1 import FaceRecognitionONet

# 人脸识别模型路径
MODEL_PATH = 'face_recognition_onet.pth'

# Haar Cascade 人脸检测器路径
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml' # 假设放在项目根目录

# DNN 人脸检测模型文件路径
# 模型的结构文件
DETECTION_PROTO_PATH = 'deploy.prototxt'
# 模型的权重文件
DETECTION_WEIGHTS_PATH = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
# 人脸检测置信度阈值
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# 模型参数 (应与训练时一致)
MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH = 48, 48

# 数据标准化参数 (应与训练时一致)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

class VideoThread(QThread):
    """
    用于在单独线程中捕获视频帧并处理。
    """
    # 定义信号，用于将处理后的帧发送回主线程
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        # self.detector = None # OpenCV DNN 人脸检测器
        self.model = None
        self.idx_to_class = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = transforms.Compose([
            transforms.Resize((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

    def run(self):
        # 捕获视频
        camera_index = 0  # 默认摄像头索引
        if isinstance(self.parent(), FaceRecognitionApp):
            camera_index = self.parent().camera_dropdown.currentData()
            if camera_index is None or camera_index == "选择摄像头...":  # 检查是否选择了有效项
                print("错误: 视频线程未获取到有效的摄像头索引。")
                self._run_flag = False
                return
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_index}。请检查摄像头是否连接或是否被其他应用占用。")
            self._run_flag = False  # 如果无法打开摄像头，停止运行
            # 可以发射一个信号通知 GUI 错误信息
            return

        # 获取视频帧的原始尺寸
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头 {camera_index} 开启，分辨率: {frame_width}x{frame_height}")
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                # 将帧从 BGR 转换为 RGB (PyTorch 通常处理 RGB)
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                # 在原始帧上绘制矩形和标签，并进行识别
                if self.model is not None and self.idx_to_class is not None:
                    for (x, y, w, h) in faces:
                        # 裁剪人脸区域
                        face_img_cv = rgb_img[y:y+h, x:x+w]

                        if face_img_cv.size > 0: # 检查裁剪区域是否有效
                            try:
                                # 将 OpenCV 图片 (numpy array) 转为 PIL Image
                                face_img_pil = Image.fromarray(face_img_cv)

                                # 预处理
                                input_tensor = self.preprocess(face_img_pil).unsqueeze(0).to(self.device)

                                # 识别 (推理)
                                with torch.no_grad():
                                    outputs = self.model(input_tensor)
                                probabilities = F.softmax(outputs, dim=1)
                                max_prob, predicted_idx_tensor = torch.max(probabilities, 1)
                                predicted_idx = predicted_idx_tensor.item()
                                confidence = max_prob.item()

                                predicted_class_name = self.idx_to_class.get(predicted_idx, "未知")

                                # 在帧上绘制矩形和标签
                                color = (255, 0, 0) # BGR 格式，这里是蓝色
                                cv2.rectangle(cv_img, (x, y), (x+w, y+h), color, 2)

                                # 标签文本
                                label_text = f"{predicted_class_name}: {confidence:.2f}"
                                # 获取文本大小以便确定放置位置
                                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                # 放置标签文本 (稍微上移一点)
                                cv2.rectangle(cv_img, (x, y - text_height - baseline), (x + text_width, y), color, -1) # 绘制背景框
                                cv2.putText(cv_img, label_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # 绘制文本

                            except Exception as e:
                                print(f"识别或绘制错误: {e}")
                                # 即使出错，也绘制检测框
                                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 0, 255), 2) # 红色框表示出错

                # 发送处理后的帧信号
                self.change_pixmap_signal.emit(cv_img)

        # 释放摄像头
        cap.release()

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

        self.image_path = None
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None

        self.video_thread = None # 视频处理线程实例

        self.initUI()
        self.load_model() # 在应用启动时加载模型

        # 初始化摄像头下拉框
        self.init_camera_dropdown()


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
        self.camera_dropdown.addItem("选择摄像头...") # 默认项
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
                self.camera_dropdown.addItem(f"摄像头 {i}", i)
                cap.release()
            else:
                # 如果连续几个索引都打不开，可能就没有更多摄像头了
                # 这里的判断可以更智能，例如连续3个打不开就停止
                pass # 暂不 break，尝试更多索引

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
            self.model = FaceRecognitionONet(num_classes=num_classes)
            self.model.load_state_dict(model_state_dict)
            self.model.to(device) # 确保模型在正确设备上
            self.model.eval() # 设置模型为评估模式
            print("识别模型加载成功。")

            # 将模型和类别映射传递给视频线程类 (在创建线程实例时设置)
            # 或者在启动视频时设置给线程实例

        except Exception as e:
            self.result_label.setText(f"错误: 加载模型失败 - {e}")
            print(f"错误: 加载模型失败 - {e}")
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

                # 将 OpenCV BGR 图片转换为 QImage (需要转换为 RGB 或灰度)
                # OpenCV images are BGR, QImage expects RGB
                height, width, channel = cv_img.shape
                bytesPerLine = 3 * width
                qimage = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_BGR888) # OpenCV 是 BGR

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


    def preprocess_image_for_recognition(self, cv_img):
        """
        对从 OpenCV 读取的人脸区域图片进行预处理，使其符合识别模型输入要求 (48x48 RGB)。
        输入是 OpenCV 格式 (BGR numpy array)。
        """
        try:
            # 将 OpenCV BGR 转换为 RGB
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            # 将 numpy array (H, W, C) 转为 PIL Image
            pil_img = Image.fromarray(rgb_img)

            # 定义与训练时一致的转换
            preprocess = transforms.Compose([
                transforms.Resize((MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH)), # 缩放到模型输入尺寸
                transforms.ToTensor(), # 转换为 Tensor ([C, H, W]), 像素值 [0, 1]
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD) # 标准化
            ])

            # 应用转换，并在批次维度上扩展 (因为模型期望批次输入)
            img_tensor = preprocess(pil_img).unsqueeze(0) # 添加批次维度 [1, C, H, W]

            # 将 tensor 移动到模型所在的设备
            device = next(self.model.parameters()).device # 获取模型当前所在的设备
            img_tensor = img_tensor.to(device)

            return img_tensor

        except Exception as e:
            print(f"图片预处理失败: {e}")
            return None

    def recognize_static_image(self):
        """
        对加载的静态图片进行人脸检测和识别。
        """
        if self.model is None or self.image_path is None:
            self.result_label.setText("模型未加载或未选择图片。")
            print("模型未加载或未选择图片，无法识别。")
            return

        if not os.path.exists(HAARCASCADE_PATH):
             self.result_label.setText(f"错误: 未找到人脸检测器文件 {HAARCASCADE_PATH}。")
             print(f"错误: 未找到人脸检测器文件 {HAARCASCADE_PATH}。")
             return

        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if face_cascade.empty():
             self.result_label.setText(f"错误: 无法加载人脸检测器文件 {HAARCASCADE_PATH}。")
             print(f"错误: 无法加载人脸检测器文件 {HAARCASCADE_PATH}。")
             return


        # 使用 OpenCV 读取原始图片 (BGR 格式)
        cv_img_original = cv2.imread(self.image_path)
        if cv_img_original is None:
            self.result_label.setText("错误: 无法读取选定的图片文件。")
            print("错误: 无法读取选定的图片文件。")
            return

        # 复制一份用于绘制结果
        cv_img_display = cv_img_original.copy()

        # 转换为灰度进行检测
        gray = cv2.cvtColor(cv_img_original, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        print("开始进行人脸检测...")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"检测到 {len(faces)} 张人脸。")

        self.result_label.setText(f"检测到 {len(faces)} 张人脸。")

        if len(faces) == 0:
             self.display_static_image(self.image_path) # 显示原图
             self.result_label.setText("未检测到人脸。")
             return

        print("开始进行人脸识别...")
        try:
            for (x, y, w, h) in faces:
                # 裁剪人脸区域 (从原始图片裁剪，确保清晰度)
                face_img_cv = cv_img_original[y:y+h, x:x+w]

                if face_img_cv.size > 0: # 检查裁剪区域是否有效
                    # 预处理人脸图片
                    input_tensor = self.preprocess_image_for_recognition(face_img_cv)
                    if input_tensor is None:
                        # 预处理失败，绘制红色框表示错误
                        cv2.rectangle(cv_img_display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        print(f"对框选区域 ({x},{y},{w},{h}) 的人脸预处理失败。")
                        continue # 跳过当前人脸


                    # 识别 (推理)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    max_prob, predicted_idx_tensor = torch.max(probabilities, 1)
                    predicted_idx = predicted_idx_tensor.item()
                    confidence = max_prob.item()

                    # 将预测索引映射回类别名称
                    predicted_class_name = self.idx_to_class.get(predicted_idx, "未知")

                    # 在复制的图片上绘制矩形和标签
                    color = (255, 0, 0) # BGR 格式，这里是蓝色
                    cv2.rectangle(cv_img_display, (x, y), (x+w, y+h), color, 2)

                    # 标签文本
                    label_text = f"{predicted_class_name}: {confidence:.2f}"
                    # 获取文本大小以便确定放置位置
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    # 放置标签文本 (稍微上移一点)
                    cv2.rectangle(cv_img_display, (x, y - text_height - baseline), (x + text_width, y), color, -1) # 绘制背景框
                    cv2.putText(cv_img_display, label_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # 绘制文本

            # 将绘制好矩形和标签的 OpenCV 图片转换为 QPixmap 显示
            height, width, channel = cv_img_display.shape
            bytesPerLine = 3 * width
            qimage = QImage(cv_img_display.data, width, height, bytesPerLine, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)

            self.display_label.setPixmap(pixmap.scaled(
                self.display_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.display_label.setText("")


        except Exception as e:
            self.result_label.setText(f"识别过程中发生错误: {e}")
            print(f"识别过程中发生错误: {e}")


    def start_video(self):
        """
        启动摄像头视频捕获和处理。
        """
        if self.model is None:
            self.result_label.setText("模型未加载，无法启动视频识别。")
            print("模型未加载，无法启动视频识别。")
            return

        if not os.path.exists(HAARCASCADE_PATH):
             self.result_label.setText(f"错误: 未找到人脸检测器文件 {HAARCASCADE_PATH}。")
             print(f"错误: 未找到人脸检测器文件 {HAARCASCADE_PATH}。")
             return

        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        if face_cascade.empty():
             self.result_label.setText(f"错误: 无法加载人脸检测器文件 {HAARCASCADE_PATH}。")
             print(f"错误: 无法加载人脸检测器文件 {HAARCASCADE_PATH}。")
             return

        # 获取选定的摄像头索引
        camera_index = self.camera_dropdown.currentData()
        if camera_index is None:
             self.result_label.setText("请选择一个有效的摄像头。")
             return

        # 检查是否已有视频线程正在运行
        if self.video_thread and self.video_thread.isRunning():
            print("视频线程已在运行。")
            return

        print(f"正在启动摄像头 {camera_index}...")
        self.video_thread = VideoThread()
        # 将模型和类别映射传递给视频线程
        self.video_thread.set_model(self.model, self.class_to_idx)
        # 将人脸检测器传递给视频线程
        self.video_thread.face_cascade = face_cascade

        # 连接信号槽，接收线程发送的帧
        self.video_thread.change_pixmap_signal.connect(self.update_image)

        # 启动线程
        self.video_thread.start()

        # 更新按钮状态
        self.select_button.setEnabled(False) # 视频运行时禁用图片选择
        self.recognize_button.setEnabled(False) # 视频运行时禁用图片识别
        self.start_video_button.setEnabled(False)
        self.stop_video_button.setEnabled(True)
        self.camera_dropdown.setEnabled(False) # 视频运行时禁用摄像头选择
        self.result_label.setText("视频捕获和识别已启动...")


    def stop_video(self):
        """
        停止摄像头视频捕获。
        """
        if self.video_thread and self.video_thread.isRunning():
            print("正在停止视频线程...")
            self.video_thread.stop()
            self.video_thread = None # 清除线程实例

            # 更新按钮状态
            self.select_button.setEnabled(True) # 停止视频后启用图片选择
            # recogniz_button 状态取决于是否有图片加载
            if self.image_path:
                 self.recognize_button.setEnabled(True)
            self.start_video_button.setEnabled(True)
            self.stop_video_button.setEnabled(False)
            self.camera_dropdown.setEnabled(True) # 停止视频后启用摄像头选择
            self.result_label.setText("视频已停止。")
            self.display_label.setText("视频已停止，请选择图片或重新启动视频。") # 清空或显示提示文本
            self.display_label.setPixmap(QPixmap()) # 清空显示的图片


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """
        接收视频线程发送的帧，并在 QLabel 中显示。
        输入是 OpenCV BGR 格式的 numpy array。
        """
        try:
            # 将 OpenCV BGR 图片转换为 QImage (需要转换为 RGB 或灰度)
            height, width, channel = cv_img.shape
            bytesPerLine = 3 * width
            qimage = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_BGR888) # OpenCV 是 BGR

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
            print(f"更新显示图片时发生错误: {e}")


    def closeEvent(self, event):
        """
        在窗口关闭时停止视频线程。
        """
        self.stop_video()
        event.accept() # 允许窗口关闭


# 应用主入口
if __name__ == '__main__':
    # 在 Windows 下使用多进程数据加载或创建线程时，最好加上 freeze_support()
    # import multiprocessing
    # multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    ex = FaceRecognitionApp()
    ex.show()
    sys.exit(app.exec_())