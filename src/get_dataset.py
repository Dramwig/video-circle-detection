import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_dataset(dataset_files, transform, kind = ''):
    dataset = []
    for txt_file, png_file in tqdm(dataset_files, desc=f"Processing {kind} dataset", total=len(dataset_files)):
        image = Image.open(png_file).convert('L')  # 转换为灰度图像
        image = transform(image)
        # 读取txt文件，每个txt包含"x,y"格式的数据
        with open(txt_file, 'r') as f:
            line = f.readline()
            x, y = map(float, line.split(','))  # 解析x和y
        dataset.append((image, torch.tensor([x, y])))  # 将图像和标签添加到数据集中
    return dataset

def pair_files(txt_files, png_files):
    paired_files = []
    # 基于文件名（无扩展名）进行匹配
    for txt_file,png_file in zip(txt_files,png_files):
        paired_files.append((txt_file, png_file))
    return paired_files

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet50 input dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single-channel
    ])
    return transform

def get_kind_dataloader(batch_size = 32, kind = 'train'):
    # 文件夹路径
    dir_data = "data"

    # 列出文件夹下的所有文件
    files_data_all = [os.path.join(dir_data, kind, file) for file in os.listdir(os.path.join(dir_data, kind))]
    
    # 分别筛选出.txt和.png文件
    files_data_txt = [file for file in files_data_all if file.endswith('.txt')]
    files_data_png = [file for file in files_data_all if file.endswith('.png')]
    
    if if_show:
        # 输出文件路径数组
        print(f"{kind} TXT文件路径：", files_data_txt[:5])
        print(f"{kind} PNG文件路径：", files_data_png[:5])

    # 创建训练、验证和测试数据集
    dataset_files = pair_files(files_data_txt, files_data_png)
    
    if if_show:
        # 输出数据集示例
        print(f"{kind} 数据集示例：", dataset_files[:3])

    # Preprocess the image  定义数据转换
    transform = get_transform()
    
    if if_show:
        # Load your single-channel image
        image_path = files_data_png[0]
        image = Image.open(image_path).convert('L')  # Convert image to grayscale

        # Display the image
        plt.imshow(image, cmap='gray')  # Use grayscale color map
        plt.show()

        image = transform(image).unsqueeze(0)  # Add batch dimension

        print(image)
        print(image.shape)
        print(image[0, 0, 80, 80])

    # 创建训练、验证和测试数据集
    dataset = read_dataset(dataset_files, transform, kind = kind)

    # 创建 DataLoader 加载训练数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if if_show:
        images, labels = next(iter(dataloader))
        print("图像形状:", images.shape)  # 应为(batch_size, channels, height, width)
        print("标签形状:", labels.shape)  # 应为(batch_size, 2)，每个样本有两个坐标值
    
    return dataloader
    
def get_dataset(batch_size = 32):
    # 获取训练、验证和测试数据集
    return get_kind_dataloader(batch_size = batch_size, kind = 'train'), get_kind_dataloader(batch_size = batch_size, kind = 'validation'), get_kind_dataloader(batch_size = batch_size, kind = 'test')
    
# Parameters
if_show = False

if __name__ == "__main__":
    # global if_show
    if_show = True
    get_dataset()