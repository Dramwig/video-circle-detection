import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
from src.module import ResNet50Regression
from src.get_dataset import get_dataset, get_kind_dataloader
from src.train import evaluate

def load_model(model_path):
    # Initialize the model
    model_load = ResNet50Regression().to(device)
    model_load.load_state_dict(torch.load(model_path))
    return model_load

def plot(output_values):
    # Plotting the line graph
    plt.figure(figsize=(8, 5))
    plt.plot(output_values, label='Output Values', marker='o')
    plt.xlabel('Image Index')
    plt.ylabel('Output Value')
    plt.title('Output Values from Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Convert output_values to a DataFrame
    folder_path = 'save'
    os.makedirs(folder_path)
    df = pd.DataFrame({'Output Values': output_values})
    excel_file_path = f'{folder_path}/output_values.xlsx'
    df.to_excel(excel_file_path, index=False)

    print(f'Output values saved to {excel_file_path}')

# Parameters
if_show = False

def test():
    global device,criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    model_load = load_model('archive/resnet50_regression_val_0.0081.pth') # 加载模型
    
    # 获取测试集上的损失值
    test_loader = get_kind_dataloader(kind = 'test')
    test_loss = evaluate(model_load, test_loader, device, criterion)
    print(f'Test loss: {test_loss}')

    # 文件夹路径
    dir_data = "data/processed"

    # 列出文件夹下的文件路径
    dir_data = [os.path.join(dir_data, file) for file in os.listdir(dir_data)]

    # 输出文件路径数组
    print("文件夹下的文件路径：", dir_data[:3])

    # Preprocess the image  定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit ResNet50 input dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for single-channel
        transforms.Lambda(lambda x: x.transpose(1, 2)) # 转置!!!用于测试y方向
    ])

    outputs = []
    model_load.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for img_dir in tqdm(dir_data,desc = 'Detection'):
            #读取图片为tensor.to(device)
            image = Image.open(img_dir).convert('L')
            image = transform(image).unsqueeze(0).to(device)
            output = model_load(image)
            outputs.append(output)
    output_values = [output.item() for output in outputs]

    plot(output_values)

    # 加载视频文件
    video_path = 'data/振动视频1.avi'
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算每帧的时间间隔
        frame_interval = 1 / fps
        print("帧率（FPS）:", fps)
        print("每帧时间间隔（秒）:", frame_interval)
    else:
        print("无法打开视频文件。")

    # 释放视频文件
    cap.release()

    # 假设 output_values 是一个numpy数组，frame_interval 是采样间隔
    # output_values = np.array([...])  # 这里填入你的数据
    # frame_interval = ...  # 每帧时间间隔

    # 执行傅立叶变换
    fft_values = np.fft.fft(output_values)

    # 计算频率轴
    n = len(output_values)  # 数据点的数量
    frequencies = np.fft.fftfreq(n, d=frame_interval)

    # 计算幅度
    magnitude = np.abs(fft_values)

    # 因为FFT输出包含负频率，我们通常只关心正频率部分
    positive_frequencies = frequencies[:n // 2]
    positive_magnitude = magnitude[:n // 2]

    # 创建DataFrame
    data = {'频率 (Hz)': positive_frequencies, '幅度': positive_magnitude}
    df = pd.DataFrame(data)

    # 保存为CSV文件
    csv_filename = 'Frequency_Magnitude_Data.csv'
    df.to_csv(csv_filename, index=False)

    print(f"数据已保存到CSV文件：{csv_filename}")


