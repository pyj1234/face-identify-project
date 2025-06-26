import cv2
import numpy as np
import os
import glob

# --- 配置参数 ---
# 模型文件路径 (请确保这些文件存在于指定路径)
prototxt_path = "./deploy.prototxt"
model_path = "./res10_300x300_ssd_iter_140000.caffemodel"

DATASET_ROOT_DIR = './dataset' # 数据集在项目文件夹下的 dataset/ 目录

# 置信度阈值
confidence_threshold = 0.5
# -----------------

def detect_and_crop_faces_dnn(image_path, prototxt_path, model_path, confidence_threshold=0.5):
    """
    使用 OpenCV DNN 人脸检测器检测图像中的人脸，并将它们裁剪出来。

    参数:
        image_path (str): 输入图像的路径。
        prototxt_path (str): Caffe模型的 .prototxt 文件的路径。
        model_path (str): Caffe模型的 .caffemodel 文件的路径。
        confidence_threshold (float): 用于过滤弱检测的置信度阈值。

    返回:
        tuple: (original_image_with_boxes, list_of_cropped_faces)
               original_image_with_boxes 是带有检测框的原始图像副本。
               list_of_cropped_faces 是一个包含裁剪出的人脸图像 (NumPy数组) 的列表。
               如果发生错误或未检测到人脸，则列表可能为空。
    """
    # 检查模型文件是否存在
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print(f"错误: 模型文件未找到。请检查路径:\nPrototxt: {prototxt_path}\nModel: {model_path}")
        return None, []

    # 加载模型
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        if net.empty():
            print("错误: 无法加载Caffe模型。")
            return None, []
    except cv2.error as e:
        print(f"OpenCV 错误: {e}")
        print("这可能是由于模型文件损坏或OpenCV DNN模块存在问题。")
        return None, []


    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None, []

    (h, w) = image.shape[:2]
    original_image_with_boxes = image.copy() # 创建副本以绘制边界框
    cropped_faces = []

    # 将图像转换为 blob
    # 模型期望的输入尺寸是 300x300
    # 均值减法参数 (104.0, 177.0, 123.0) 是模型训练时使用的
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 设置输入并进行前向传播
    net.setInput(blob)
    detections = net.forward() # detections.shape 通常是 (1, 1, N, 7)

    # 遍历检测结果
    for i in range(0, detections.shape[2]):
        # 提取当前检测的置信度 (概率)
        confidence = detections[0, 0, i, 2]

        # 过滤掉置信度低的检测
        if confidence > confidence_threshold:
            # 计算边界框的 (x, y) 坐标
            # 边界框坐标是归一化的，需要乘以原始图像的尺寸
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 确保边界框在图像范围内
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # 检查裁剪区域是否有效
            if startX < endX and startY < endY:
                # 裁剪人脸
                face = image[startY:endY, startX:endX]
                cropped_faces.append(face)

                # 在带框的图像上绘制边界框和置信度
                text = f"{confidence*100:.2f}%"
                y_text = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(original_image_with_boxes, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(original_image_with_boxes, text, (startX, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if not cropped_faces:
        print("未检测到符合置信度阈值的人脸。")

    return original_image_with_boxes, cropped_faces

# --- 主程序 ---
if __name__ == "__main__":
    detected_num = 0 # 用于计数检测到的人脸数量

    classes = sorted([d for d in os.listdir(DATASET_ROOT_DIR) if os.path.isdir(os.path.join(DATASET_ROOT_DIR, d)) and d.startswith('s')],
                                key=lambda x: int(x[1:])) # 按 s 后面的数字排序
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    for cls in classes:
        cls_path = os.path.join(DATASET_ROOT_DIR, cls)
        pgm_files = glob.glob(os.path.join(cls_path, '*.pgm'))
        if not pgm_files:
            print(f"警告: 类别 {cls} ({cls_path}) 下没有找到 .pgm 图片文件，该类别将被跳过。")
            continue # 跳过没有图片的类别

        for image_path in pgm_files:
            # 确保图片存在
            if not os.path.exists(image_path):
                print(f"错误: 输入图像 '{image_path}' 未找到。请创建一个名为 {image_path} 的图片文件，或修改 image_path 变量。")
            else:
                print(f"正在处理图像: {image_path}")

            # 调用函数进行检测和裁剪
            image_with_boxes, faces = detect_and_crop_faces_dnn(image_path, prototxt_path, model_path, confidence_threshold)

            if image_with_boxes is not None:
                detected_num += 1
            #     # 显示带有检测框的原始图像
            #     cv2.imshow("Image with Detections", image_with_boxes)
            #     cv2.waitKey(1) # 等待1ms，确保窗口有时间刷新

            # if faces:
            #     print(f"成功检测并裁剪出 {len(faces)} 张人脸。")
            #     # 显示裁剪出的人脸
            #     for i, face_img in enumerate(faces):
            #         if face_img.size > 0: # 确保图像不是空的
            #             cv2.imshow(f"Cropped Face {i+1}", face_img)
            #             cv2.waitKey(1) # 等待1ms
            #         else:
            #             print(f"警告: 裁剪出的人脸 {i+1} 是空的。")
            #     print("按任意键关闭所有窗口...")
            #     cv2.waitKey(0) # 等待按键后关闭
            # else:
            #     if image_with_boxes is not None: # 如果图像加载成功但未检测到人脸
            #         print("在图像中未检测到人脸，或置信度过低。")
            #         print("按任意键关闭窗口...")
            #         cv2.waitKey(0)


            # cv2.destroyAllWindows()
    print(f"检测到 {detected_num} 张人脸。")