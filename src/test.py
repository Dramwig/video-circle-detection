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
from src.module import UserModule as UserModule
from src.get_dataset import get_dataset, get_kind_dataloader
from src.train import evaluate
from src.config import save_folder_path

def load_model(model_path):
    # Initialize the model
    model_load = UserModule().to(device)
    model_load.load_state_dict(torch.load(model_path))
    return model_load

def plot(output_values, str_index = ''):
    # Plotting the line graph
    if if_show:
        plt.figure(figsize=(8, 5))
        plt.plot(output_values, label='Output Values', marker='o')
        plt.xlabel('Image Index')
        plt.ylabel('Output Value')
        plt.title('Output Values from Model')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Convert output_values to a DataFrame
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    df = pd.DataFrame({'Output Values': output_values})
    excel_file_path = f'{save_folder_path}/output_values{str_index}.xlsx'
    df.to_excel(excel_file_path, index=False)

    print(f'Output values saved to {excel_file_path}')
    
def fft(output_values, str_index = ''):
    # 加载视频文件
    video_path = 'data/振动视频1.avi'
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算每帧的时间间隔
        frame_interval = 1 / fps
        if if_show:
            print("帧率（FPS）:", fps)
            print("每帧时间间隔（秒）:", frame_interval)
    else:
        print("无法打开视频文件。")

    # 释放视频文件
    cap.release()

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
    csv_filename = f'{save_folder_path}/Frequency_Magnitude_Data{str_index}.csv'
    df.to_csv(csv_filename, index=False)

    print(f"FFT values saved to {csv_filename}")

# Parameters
if_show = False

def test(dataset_files = None, str_index = '', model_load_path = 'archive/resnet50_regression_val_0.0081.pth', orientations = 'y'):
    global device,criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss() # 定义损失函数
    try:
        model_load = load_model(model_load_path) # 加载模型
    except:
        print('Model not found')
        return
    
    # 获取测试集上的损失值
    if if_show:
        test_loader = get_kind_dataloader(kind = 'test')
        test_loss = evaluate(model_load, test_loader, device, criterion)
        print(f'Test loss: {test_loss}')
    
    if dataset_files is None: # 加载数据集
        dir_data = f"data/processed{str_index}" # 文件夹路径
        dataset_files = [os.path.join(dir_data, file) for file in os.listdir(dir_data)] # 列出文件夹下的文件路径

    # 定义数据转换
    if orientations == 'y':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit ResNet50 input dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for single-channel
            transforms.Lambda(lambda x: x.transpose(1, 2)) # 转置!!!用于测试y方向
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit ResNet50 input dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single-channel
        ])

    outputs = []
    model_load.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for img_dir in tqdm(dataset_files,desc = 'Detection'):
            #读取图片为tensor.to(device)
            image = Image.open(img_dir).convert('L')
            image = transform(image).unsqueeze(0).to(device)
            output = model_load(image)
            outputs.append(output)
    output_values = [output.item() for output in outputs]

    plot(output_values, str_index)
    fft(output_values, str_index)

if __name__ == '__main__':
    # for i in range(1, 6):
    #     test(str_index = str(i))
    if_show = True
    test()


