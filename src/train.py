import pandas as pd
import torch
from tqdm import tqdm
import os
import shutil
import matplotlib.pyplot as plt
from src.get_dataset import get_dataset
from src.module import ResNet50Regression
from torch import nn

def evaluate(model, test_loader, device, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        test_loader_tqdm = tqdm(test_loader, desc=f'Evaluate')
        for images, targets in test_loader_tqdm:  # 使用测试数据集
            images = images.to(device)
            targets = targets.to(device)

            # 执行前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, targets[:, 0].unsqueeze(1))
            total_loss += loss.item()

    # 计算平均损失并返回
    return total_loss / len(test_loader)

def show_losses(train_losses, val_losses):
    # Create a DataFrame from the lists
    losses_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    })
    csv_file_path = 'save/losses.csv'  # You can change this to your desired path and filename
    losses_df.to_csv(csv_file_path, index=False) # Save the DataFrame to a CSV file
    print(f'Losses saved to {csv_file_path}')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 设置 y 轴的范围，例如从 0 到 1
    plt.ylim(0, 1)
    plt.show()
    
def clear_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 清空文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        # 如果文件夹不存在，则创建它
        os.makedirs(folder_path)
    

def train(num_epochs = 500, batch_size = 32):
    # Get the dataset loaders
    train_loader, validation_loader, test_loader = get_dataset(batch_size = batch_size)
    
    # Define the device based on the availability of CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use {device}')

    # Initialize the model and move it to the chosen device
    model = ResNet50Regression().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) # 学习率调整策略

    # 定义文件夹路径
    save_folder_path = 'save'
    clear_folder(save_folder_path)
        
    # Training process
    train_losses = []
    val_losses = []
    min_val_lost = 99999999

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        # Wrap the train_loader with tqdm for a progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, targets in train_loader_tqdm:
            # Move images and targets to the chosen device
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)
            loss = criterion(outputs, targets[:,0].unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Print average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(model, validation_loader, device, criterion)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 每个epoch后更新学习率
        scheduler.step()
        param_group = optimizer.param_groups[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}, Current learning rate: {param_group["lr"]:.5f}')

        if min_val_lost > avg_val_loss:
            min_val_lost = avg_val_loss
            model_name = f'{save_folder_path}/resnet50_regression_val_{min_val_lost:.4f}.pth'  # 使用损失值作为文件名的一部分
            torch.save(model.state_dict(), model_name)
        
    # 获取测试集上的损失值
    test_loss = evaluate(model, test_loader, device, criterion)
    print(f'Test loss: {test_loss}')

    # 保存模型
    if os.path.exists('archive') == False:
        os.makedirs('archive')
    model_name = f'archive/resnet50_regression_test_{test_loss:.4f}.pth'  # 使用损失值作为文件名的一部分
    torch.save(model.state_dict(), model_name)

    show_losses(train_losses, val_losses)
    
    return model_name

if __name__ == '__main__':
    train()