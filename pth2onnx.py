import torch
from torchvision import transforms
import os
import onnx
import onnxruntime
from Res2Net_v1b import res2net50_v1b_26w_4s
from model2 import FaceIDResNetV1

def export_to_onnx(model, input_size, output_path):
    """
    导出 PyTorch 模型为 ONNX 格式
    :param model: PyTorch 模型
    :param input_size: 输入尺寸 (C, H, W)
    :param output_path: 输出 ONNX 文件路径
    """
    # 创建一个示例输入张量
    dummy_input = torch.randn(1, *input_size)

    # 导出模型
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,
        output_path,  # 导出路径
        export_params=True,  # 导出模型参数
        opset_version=11,  # ONNX操作集版本
        do_constant_folding=True,  # 执行常量折叠优化
        input_names=['input'],  # 输入张量名称
        output_names=['output'],  # 输出张量名称
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'Height', 3: 'Width'},  # 动态轴
            'output': {0: 'batch_size'}
        }
    )

if __name__ == '__main__':
    num_classes_example = 40
    input_path = 'param/face_recognition_onet.pth'
    output_path = 'param/face_recognition_onet.onnx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceIDResNetV1(resnet=res2net50_v1b_26w_4s(pretrained=True), num_classes=num_classes_example)
    if os.path.exists(input_path):
        try:
            checkpoint = torch.load(input_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载成功")
        except Exception as e:
            print(f"错误: 加载模型失败 - {e}")
            exit(1)

    # 设置输入尺寸 (C, H, W)
    input_size = (3, 48, 48)

    # 导出模型到 ONNX 格式
    export_to_onnx(model, input_size, output_path)

    print(f"模型已导出到 {output_path}")
