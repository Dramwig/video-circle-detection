from PIL import Image
import os
from src.frame_processing import display
import torch
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import numpy as np
import random
from tqdm import tqdm

def check_image():
    dir_processed_data = [os.path.join("data/processed", file) for file in os.listdir("data/processed")] # 列出文件夹下的文件路径     # 列出文件夹下的文件路径    

    # 假设您已经导入了Image和display模块，并定义了file变量和image对象
    file = dir_processed_data[0]
    image = Image.open(file)
    display(image)

    # 将图像转换为张量
    to_tensor = ToTensor()
    image_tensor = to_tensor(image)

    # 显示张量的形状和数据类型
    print("图像张量的形状：", image_tensor.shape)
    print("图像张量的数据类型：", image_tensor.dtype)

    # 获取图像张量的形状
    channels, height, width = image_tensor.shape

    # 假设要获取像素位置为(row, col)的像素值
    row, col = (int)(128/3), (int)(128/3)  # 举例位置，您可以根据实际需求修改
    gray_value = image_tensor[0, row, col].item()
    print(f"像素({row}, {col})的灰度值为：{gray_value}")

    # 统计值为1的像素个数
    num_pixels_equal_to_1 = torch.sum(image_tensor == 1).item()
    print("值为1的像素个数：", num_pixels_equal_to_1)

    # 统计值为0的像素个数
    num_pixels_equal_to_0 = torch.sum(image_tensor == 0).item()
    print("值为0的像素个数：", num_pixels_equal_to_0)


def add_irregular_smudge(image, draw, width=128, height=128):
    center = (random.uniform(0, width), random.uniform(0, height))  # 污迹的中心位置
    max_radius = 10  # 污迹最大半径
    steps = 60  # 污迹形成的步骤数
    x_center, y_center = center
    width, height = image.size

    theta = random.uniform(0, 2 * np.pi)
    
    for _ in range(steps):
        angle = random.gauss(theta, np.pi/4)
        radius = random.uniform(0, max_radius)
        x = int(x_center + radius * np.cos(angle))
        y = int(y_center + radius * np.sin(angle))

        # 在选定的点周围画小圆以模拟污迹的不规则形状
        for dy in range(-3, 4):  # 这里的数字可以根据需要调整以改变污迹的密集度
            for dx in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    draw.point((nx, ny), fill=0)  # 使用0表示黑色


def create_image(width, height, background_color):
    image = Image.new("L", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    return image, draw

def draw_circular_ring(draw, width, height, inner_radius, outer_radius):
    focus1, focus2 = get_foci(width, height)
    for y in range(height):
        for x in range(width):
            distance_to_center = np.sqrt((x - focus1[0]) ** 2 + (y - focus1[1]) ** 2) + \
                                 np.sqrt((x - focus2[0]) ** 2 + (y - focus2[1]) ** 2)
            if 2 * inner_radius <= distance_to_center <= 2 * outer_radius:
                draw.point((x, y), fill=0)  # Black
    return focus1, focus2

def add_smudges(draw, width, height, inner_radius, outer_radius, gross_point_width, gross_point_width2, focus1, focus2):
    direction = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for y in range(height):
        for x in range(width):
            distance_to_center = np.sqrt((x - focus1[0]) ** 2 + (y - focus1[1]) ** 2) + \
                                 np.sqrt((x - focus2[0]) ** 2 + (y - focus2[1]) ** 2)
            # 在圆环边缘添加一定程度的污迹
            if 2*inner_radius - gross_point_width <= distance_to_center <= 2*inner_radius + gross_point_width or \
                 2*outer_radius - gross_point_width <= distance_to_center <= 2*outer_radius + gross_point_width:
                if random.random() < 0.05:  # 控制污迹生成的概率
                    for dx,dy in direction:
                        draw.point((x+dx, y+dy), fill=0)  # 使用0表示黑色
                if random.random() < 0.05:  # 控制污迹生成的概率
                    for dx,dy in direction:
                        draw.point((x+dx, y+dy), fill=255)  
            if 2*inner_radius - gross_point_width2 <= distance_to_center <= 2*inner_radius + gross_point_width2 or \
                 2*outer_radius - gross_point_width2 <= distance_to_center <= 2*outer_radius + gross_point_width2:
                if random.random() < 0.1:  
                    draw.point((x, y), fill=0)  
                if random.random() < 0.1:  
                    draw.point((x, y), fill=255)  

def add_noise(image, draw, width, height, radio):
   # 随机选取点并随机翻转颜色
    num_points = int(width * height * radio)  # 随机选取的点的数量，可根据需要调整
    points = [(random.randint(0, width - 1), random.randint(0, height - 1)) for _ in range(num_points)]

    for point in points:
        x, y = point
        current_color = image.getpixel((x, y))
        new_color = 255 - current_color  # 黑色变为白色，白色变为黑色
        draw.point((x, y), fill=new_color)

def get_foci(width, height):
    random_displacement_range = width / 10
    mu, sigma = 0, 5  # Mean and standard deviation for Gaussian distribution
    focus1 = (width / 2 + random.uniform(-random_displacement_range, random_displacement_range), 
              height / 2 + random.uniform(-random_displacement_range, random_displacement_range))
    focus2 = (focus1[0] + random.gauss(mu, sigma), focus1[1] + random.gauss(mu, sigma))
    return focus1, focus2

def smooth_noise(image, draw, width, height, num_0_change_to_0):
    for y in range(height):
        for x in range(width):
            # Count the number of adjacent pixels that are black
            num_0 = sum(image.getpixel((nx, ny)) == 0
                        for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                        if 0 <= nx < width and 0 <= ny < height)

            # Update the pixel color based on the number of adjacent black pixels and the specified probabilities
            fill_color = 0 if random.random() < num_0_change_to_0[num_0] else 255
            draw.point((x, y), fill=fill_color)
            
def try_data_generator():
    random_number_near_one = random.uniform(0.8, 1.2)
    inner_radius = 18*2 * random_number_near_one
    outer_radius = 40*2 * random_number_near_one * random.uniform(0.9, 1.1)

    image, draw = create_image(width, height, background_color)
    focus1, focus2 = draw_circular_ring(draw, width, height, inner_radius, outer_radius)
    display(image)
    add_smudges(draw, width, height, inner_radius, outer_radius, gross_point_width, gross_point_width2,  focus1, focus2)
    display(image)
    add_noise(image, draw, width, height, 0.01)
    display(image)
    add_irregular_smudge(image, draw)
    add_irregular_smudge(image, draw)
    add_irregular_smudge(image, draw)
    display(image)
    smooth_noise(image, draw, width, height, num_0_change_to_0)
    display(image)

    center = ((focus1[0] + focus2[0])/2 ,(focus1[1] + focus2[1])/2)
    print(center)

def save_dataset( i, image, center, base_dir, set_name):
    set_dir = os.path.join(base_dir, set_name)
    if not os.path.exists(set_dir):
        os.makedirs(set_dir)
        
    image_path = os.path.join(set_dir, f'image_{i:05}.png')
    center_path = os.path.join(set_dir, f'center_{i:05}.txt')

    # 保存图像
    image.save(image_path)

    # 保存中心坐标到文本文件
    with open(center_path, 'w') as f:
        f.write(f'{center[0]},{center[1]}')

def generate_image_and_center(i, set_name):
    # Parameters for the image
    random_number_near_one = random.uniform(0.8, 1.2)
    inner_radius = 18*2 * random_number_near_one
    outer_radius = 40*2 * random_number_near_one * random.uniform(0.9, 1.1)

    # Create image
    image, draw = create_image(width, height, background_color)
    
    # Add shapes and effects
    focus1, focus2 = draw_circular_ring(draw, width, height, inner_radius, outer_radius)
    center = ((focus1[0] + focus2[0])/2, (focus1[1] + focus2[1])/2)
    save_dataset(i, image, center, dir_temp, set_name)
    add_smudges(draw, width, height, inner_radius, outer_radius, gross_point_width, gross_point_width2, focus1, focus2)
    save_dataset(i+1, image, center, dir_temp, set_name)
    add_noise(image, draw, width, height, 0.01)
    save_dataset(i+2, image, center, dir_temp, set_name)
    for _ in range(random.randint(1, 4)):
        add_irregular_smudge(image, draw)
    save_dataset(i+3, image, center, dir_temp, set_name)
    smooth_noise(image, draw, width, height, num_0_change_to_0)
    save_dataset(i+4, image, center, dir_temp, set_name)


# Parameters
width, height = 224, 224
dir_temp = "data"
background_color = "white"
gross_point_width = 0.5
gross_point_width2 = 2
num_0_change_to_0 = [0.0001, 0.00045, 0.7, 0.9998, 0.9999]
    
def data_generator(train_size=6000, test_size=2000, validation_size=2000):
    # 生成训练集
    for i in tqdm(range(0, train_size, 5), desc="Generating train dataset"):
        generate_image_and_center(i, "train")
    # 生成测试集
    for i in tqdm(range(0, test_size, 5), desc="Generating test dataset"):
        generate_image_and_center(i, "test")
    # 生成验证集
    for i in tqdm(range(0, validation_size, 5), desc="Generating validation dataset"):
        generate_image_and_center(i, "validation")


if __name__ == "__main__":
    # check_image()
    # try_data_generator()
    
    # 生成训练集
    data_generator(20000,2000,2000)



