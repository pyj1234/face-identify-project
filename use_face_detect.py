import cv2
import numpy as np
import os
import glob

# --- 配置参数 ---
# 模型文件路径
prototxt_path_global = "./deploy.prototxt"
model_path_global = "./res10_300x300_ssd_iter_140000.caffemodel"

DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录
OUTPUT_DIR = './extracted_faces' # 用于保存提取的人脸的目录

# 置信度阈值
confidence_threshold_global = 0.5
# 目标人脸尺寸
TARGET_FACE_SIZE = (224, 224) # (宽度, 高度)
# -----------------

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_highest_confidence_face_resized(
    image_data,
    prototxt_path=prototxt_path_global,
    model_path=model_path_global,
    target_size=TARGET_FACE_SIZE,
    overall_confidence_threshold=confidence_threshold_global
):
    """
    使用 OpenCV DNN 检测器从输入图像中提取所有人脸，
    并将其调整到指定尺寸。

    参数:
        image_data (cv2.Mat): 输入图像的 NumPy 数组 (cv2.Mat)。
        prototxt_path (str): Caffe模型的 .prototxt 文件的路径。
        model_path (str): Caffe模型的 .caffemodel 文件的路径。
        target_size (tuple): (宽度, 高度) 用于调整裁剪出的人脸图像的尺寸。
        overall_confidence_threshold (float): 只有当检测到的最高置信度高于此阈值时，才返回人脸。

    返回:
        np.ndarray: 调整大小后的单个人脸图像，如果未找到符合条件的人脸则返回 None。
    """
    # 1. 检查模型文件是否存在
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"错误: 模型文件未找到。\nPrototxt: {prototxt_path}\nModel: {model_path}")
        return None

    # 2. 加载模型
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        if net.empty():
            print("错误: 无法加载Caffe模型。")
            return None
    except cv2.error as e:
        print(f"OpenCV DNN模型加载错误: {e}")
        return None

    # 3. 加载图像
    image = image_data

    (h, w) = image.shape[:2]
    if h == 0 or w == 0:
        print("错误: 输入图像尺寸无效。")
        return None

    # 4. 将图像转换为 blob 并进行前向传播
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False) # swapRB=False for Caffe
    net.setInput(blob)
    detections = net.forward()

    # 5. 寻找置信度满足要求的人脸
    best_box_coords = None

    all_detect_faces = []
    all_detect_face_images = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > overall_confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 确保边界框有效且在图像范围内
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            width = endX - startX
            height = endY - startY

            if startX < endX and startY < endY:
                all_detect_faces.append((startX, startY, width, height))

                face_crop = image[startY:endY, startX:endX]
                if face_crop.size > 0:
                    resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
                    # print(f"  成功提取人脸，置信度: {max_confidence_score:.2f}")
                    all_detect_face_images.append(resized_face)
                else:
                    # print(f"  警告: 裁剪出的最高置信度人脸区域为空 (置信度: {max_confidence_score:.2f})。")
                    all_detect_face_images.append(None)
    return all_detect_face_images, all_detect_faces
