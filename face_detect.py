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
    image_input,
    prototxt_path=prototxt_path_global,
    model_path=model_path_global,
    target_size=TARGET_FACE_SIZE,
    overall_confidence_threshold=confidence_threshold_global
):
    """
    使用 OpenCV DNN 检测器从输入图像中提取置信度最高的人脸，
    并将其调整到指定尺寸。

    参数:
        image_input (str or np.ndarray): 输入图像的路径或已加载的图像数据 (NumPy 数组)。
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
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            print(f"错误: 无法读取图像 {image_input}")
            return None
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy() # 在副本上操作
    else:
        print("错误: image_input 必须是有效的图像路径 (str) 或 NumPy 数组。")
        return None

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

    # 5. 寻找置信度最高的人脸
    max_confidence_score = -1.0
    best_box_coords = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence_score:
            max_confidence_score = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 确保边界框有效且在图像范围内
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            if startX < endX and startY < endY:
                best_box_coords = (startX, startY, endX, endY)

    # 6. 如果找到了人脸并且其置信度满足阈值，则裁剪并调整大小
    if best_box_coords is not None and max_confidence_score >= overall_confidence_threshold:
        (startX, startY, endX, endY) = best_box_coords
        face_crop = image[startY:endY, startX:endX]

        if face_crop.size > 0:
            resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            # print(f"  成功提取人脸，置信度: {max_confidence_score:.2f}")
            return resized_face
        else:
            # print(f"  警告: 裁剪出的最高置信度人脸区域为空 (置信度: {max_confidence_score:.2f})。")
            return None
    else:
        # if best_box_coords is not None: # 意味着检测到了人脸，但置信度不够高
        #     print(f"  检测到的最高置信度 ({max_confidence_score:.2f}) 未达到阈值 {overall_confidence_threshold}。")
        # else: # 意味着根本没有检测到任何东西
        #     print("  未检测到任何人脸。")
        return None

# --- 主程序 ---
if __name__ == "__main__":
    successful_extractions = 0 # 用于计数成功提取到人脸的图像数量
    total_images_processed = 0

    # 检查数据集根目录是否存在
    if not os.path.isdir(DATASET_ROOT_DIR):
        print(f"错误: 数据集根目录 '{DATASET_ROOT_DIR}' 未找到。请确保路径正确。")
        exit()

    classes = sorted([d for d in os.listdir(DATASET_ROOT_DIR) if os.path.isdir(os.path.join(DATASET_ROOT_DIR, d)) and d.startswith('s')],
                     key=lambda x: int(x[1:])) # 按 s 后面的数字排序

    if not classes:
        print(f"警告: 在 '{DATASET_ROOT_DIR}' 中没有找到符合 's*' 格式的类别子目录。")
        exit()

    for cls_idx, cls_name in enumerate(classes):
        if successful_extractions >= 1: # 只提取1个人的脸
            break
        cls_path = os.path.join(DATASET_ROOT_DIR, cls_name)
        # 支持 .pgm, .jpg, .jpeg, .png
        image_files = []
        for ext in ('*.pgm', '*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob.glob(os.path.join(cls_path, ext)))

        if not image_files:
            print(f"警告: 类别 {cls_name} ({cls_path}) 下没有找到支持的图片文件，该类别将被跳过。")
            continue

        print(f"\n正在处理类别: {cls_name} ({len(image_files)} 张图片)")

        for image_path in image_files:
            total_images_processed += 1
            # print(f"处理图像: {image_path}")

            # 调用函数提取单个最高置信度的人脸并调整大小
            extracted_face = extract_highest_confidence_face_resized(
                image_path,
                prototxt_path=prototxt_path_global,
                model_path=model_path_global,
                target_size=TARGET_FACE_SIZE,
                overall_confidence_threshold=confidence_threshold_global
            )

            if extracted_face is not None:
                successful_extractions += 1
                # print(f"  成功提取并调整人脸大小，保存到 {OUTPUT_DIR}")

                # 显示提取到的人脸
                # cv2.imshow(f"Extracted Face from {os.path.basename(image_path)}", extracted_face)
                # key = cv2.waitKey(100) # 显示100ms，按任意键会立即跳到下一张或结束
                # if key == 27: # ESC键
                #     print("用户中断。")
                #     cv2.destroyAllWindows()
                #     exit()

                # 保存提取到的人脸
                output_filename = f"{cls_name}_{os.path.splitext(os.path.basename(image_path))[0]}_face.png"
                output_save_path = os.path.join(OUTPUT_DIR, output_filename)
                try:
                    cv2.imwrite(output_save_path, extracted_face)
                    # print(f"    已保存: {output_save_path}")
                except Exception as e:
                    print(f"    保存失败 {output_save_path}: {e}")
            else:
                print(f"  在图像 {os.path.basename(image_path)} 中未提取到符合条件的人脸。")
                pass # 之前函数内部已经打印了信息

    print(f"\n--- 处理完成 ---")
    print(f"总共处理图像数量: {total_images_processed}")
    print(f"成功提取并调整大小的人脸数量: {successful_extractions}")

    # if successful_extractions > 0:
    #     print(f"提取的人脸图像已调整为 {TARGET_FACE_SIZE} 尺寸。")
    #     print(f"如果取消注释了保存代码，文件应保存在: {OUTPUT_DIR}")

    # cv2.destroyAllWindows() #确保所有窗口都关闭