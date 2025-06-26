import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import glob
from collections import defaultdict


# 定义一个函数用于将1通道图像转换为3通道
def repeat_channel(x):
    """Repeats a single channel tensor across 3 channels."""
    if x.size(0) == 1:
        return x.repeat(3, 1, 1)
    return x

class PGMDataset(Dataset):
    def __init__(self, data_list, class_to_idx, transform=None):
        """
        自定义 PGM 图片数据集。

        参数:
            data_list (list): 包含 (image_path, label) 元组的列表。
            class_to_idx (dict): 类别名称到索引的映射字典。
            transform (callable, optional): 应用于图片的转换。
        """
        self.data_list = data_list
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = list(class_to_idx.keys()) # 用于评估报告等地方

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]

        # 使用 Pillow 读取 PGM 图片
        try:
            # PGM 通常是灰度图 'L' 模式
            img = Image.open(img_path).convert('L')
            # 将灰度图转换为 RGB，因为模型期望3通道输入
            img = img.convert('RGB')
        except Exception as e:
            print(f"警告: 无法读取或处理图片: {img_path}, 错误: {e}. 使用空白图片代替。")
            # 返回一个空白图片作为示例，避免DataLoader中断
            img = Image.new('RGB', (112, 112), color = 'black')
            # 更好的做法是跳过这个样本，但这需要更复杂的 collate_fn 或在 __init__ 中过滤无效路径

        if self.transform:
            img = self.transform(img)

        return img, label

def load_and_split_dataset(root_dir, train_split_ratio=0.3, img_height=112, img_width=112):
    """
    加载数据集，获取图片路径和标签，并按比例划分训练集和测试集的数据列表。

    参数:
        root_dir (str): 数据集的根目录 (例如 'dataset/').
        train_split_ratio (float): 训练集占总数据的比例 (0.0 到 1.0).
        img_height (int): 图片高度，用于数据转换。
        img_width (int): 图片宽度，用于数据转换。

    返回:
        tuple: 包含 (train_dataset, test_dataset, class_to_idx) 的元组，它们是 PGMDataset 的实例和类别映射。
               返回 None, None, None 如果找不到数据。
    """
    train_subset_list = []
    test_subset_list = []
    # 确保子文件夹名称按数字顺序排序，以保证类别映射的稳定性和可复现性
    class_names_sorted = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('s')],
                                key=lambda x: int(x[1:])) # 按 s 后面的数字排序
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names_sorted)}
    # idx_to_class = {i: class_name for class_name, i in class_to_idx.items()}

    if not class_names_sorted:
        print(f"错误: 在目录 {root_dir} 中未找到以 's' 开头且包含数字的子文件夹。")
        return None, None, None

    print(f"检测到 {len(class_names_sorted)} 个类别: {class_names_sorted}")

    # 遍历子文件夹查找图片
    for class_name in class_names_sorted:
        class_dir = os.path.join(root_dir, class_name)
        # glob 查找所有 .pgm 文件 (大小写不敏感)
        pgm_files = glob.glob(os.path.join(class_dir, '*.pgm')) + glob.glob(os.path.join(class_dir, '*.PGM'))

        if not pgm_files:
             print(f"警告: 类别 {class_name} ({class_dir}) 下没有找到 .pgm 图片文件，该类别将被跳过。")
             continue # 跳过没有图片的类别

        full_data_list = [] # 用于存储当前类别的所有图片路径和标签

        label = class_to_idx[class_name]
        cur_class_pgm_size = len(pgm_files)
        train_size = int(cur_class_pgm_size * train_split_ratio)
        test_size = cur_class_pgm_size - train_size
        for img_path in pgm_files:
            full_data_list.append((img_path, label)) # 直接存储路径和标签

        # 构建完整的数据列表 (路径和标签)
        if len(full_data_list) == 0:
            print("错误: 数据列表为空。")
            return None, None, None
        
        # 这里 random_split 返回的是 Subset 对象
        train_subset, test_subset = random_split(full_data_list, [train_size, test_size])
        train_subset_list.extend(train_subset)
        test_subset_list.extend(test_subset)

    if not train_subset_list or not test_subset_list:
        print(f"错误: 在目录 {root_dir} 的子文件夹中未找到任何 .pgm 图片文件。")
        return None, None, None
    print(f"总共找到 {len(train_subset_list) + len(test_subset_list)} 张图片。")

    # 定义数据转换
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)), # 缩放图片
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 转换为 Tensor ([C, H, W]), 像素值 [0, 1]
        # 灰度转RGB后ToTensor是3通道，这里不需要 repeat_channel
        # transforms.Lambda(repeat_channel), # 如果需要在 ToTensor 后处理 (现在不需要)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化 (使用 ImageNet 均值和标准差)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
         # 灰度转RGB后ToTensor是3通道，这里不需要 repeat_channel
        # transforms.Lambda(repeat_channel), # 如果需要在 ToTensor 后处理 (现在不需要)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建 PGMDataset 实例
    # 当前 PGMDataset 接受 (path, label) 的 list，所以从 Subset 中提取
    train_data_list = [train_subset_list[i] for i in range(len(train_subset_list))]
    test_data_list = [test_subset_list[i] for i in range(len(test_subset_list))]

    # 在创建 PGMDataset 实例时应用 transform
    train_dataset = PGMDataset(train_data_list, class_to_idx, transform=train_transforms)
    test_dataset = PGMDataset(test_data_list, class_to_idx, transform=test_transforms)

    # 将 class_to_idx 也返回，以便后续使用 num_classes
    return train_dataset, test_dataset, class_to_idx

if __name__ == '__main__':
    # 假设你的数据集在项目根目录下的 'dataset' 文件夹
    dataset_root = './dataset'
    train_ds, test_ds, class_mapping = load_and_split_dataset(dataset_root)

    if train_ds and test_ds and class_mapping:
        print("\n数据加载和划分成功:")
        print(f"训练集样本数: {len(train_ds)}")
        print(f"测试集样本数: {len(test_ds)}")
        print(f"类别到索引的映射: {class_mapping}")
        # 注意：Subset 对象没有 .classes 属性，需要通过 class_mapping 获取
        print(f"类别名称列表 (从 class_to_idx 生成): {list(class_mapping.keys())}")

        # 示例 DataLoader
        # 设置 num_workers=0 可以在当前主进程中加载，避免多进程问题进行初步测试
        # 如果 num_workers>0 仍然报错，可能是其他多进程兼容性问题
        train_loader_example = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
        test_loader_example = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

        # 获取一个批次的数据示例
        print("\n获取一个批次的数据示例:")
        try:
            images, labels = next(iter(train_loader_example))
            print(f"图片批次形状: {images.shape}") # 应该是 [batch_size, 3, H, W]
            print(f"标签批次形状: {labels.shape}") # 应该是 [batch_size]
            print(f"标签示例: {labels}")
        except StopIteration:
            print("DataLoader为空，无法获取批次。")
        except Exception as e:
            print(f"获取批次时发生错误: {e}")
