import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm

def pasue():
    if if_pause:
        input("Press Enter to continue...")

def display(image):
    # 显示图片
    if if_check:
        image.show()
def display_dir(image_dir):
    # 显示图片
    if if_check:
        image = Image.open(image_dir) # 读取BMP图片
        image.show()

def process_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True) # 创建输出目录
    cap = cv2.VideoCapture(video_path) # 打开视频文件
    frame_count = 0 # 初始化帧计数器
    # 循环直到视频结束
    while True:
        ret, frame = cap.read() # 读取一帧
        if not ret: # 检查是否成功读取了一帧
            break
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.bmp')  # 构建输出图片的路径
        cv2.imwrite(frame_path, frame) # 保存帧为BMP格式
        frame_count += 1 # 更新帧计数器

    cap.release() # 释放视频捕获对象
    print(f'All frames are saved in {output_dir}. Total frames: {frame_count}')
    
    files_data = [os.path.join(output_dir, file) for file in os.listdir(output_dir)] # 列出文件夹下的文件路径    
    print("文件夹下的文件路径：", files_data[:3]) # 输出文件路径数组的前三个元素
    display_dir(files_data[0])  # 显示图片
    pasue()  # 暂停
    return files_data
        
def process_cropped(input_data_dir, output_dir, output_width, output_height,):
    print(input_data_dir[0])  # 打印路径，检查是否正确
    img = cv2.imread(input_data_dir[0])

    #pts1 = np.float32([[113,229],[160,229],[113,274],[159,275]])
    #pts1 = np.float32([[257,228],[302,227],[257,272],[302,272]])
    #pts1 = np.float32([[449,223],[494,223],[449,268],[494,268]])
    #pts1 = np.float32([[647,222],[689,222],[647,265],[688,265]])
    #pts1 = np.float32([[825,220],[865,220],[825,262],[865,262]])
    pts1 = np.float32([[981,218],[1019,216],[981,259],[1019,257]])

    pts2 = np.float32([[0,0],[output_width,0],[0,output_height],[output_width,output_height]])
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width,height))
    
    for x in range(0, 4):
        cv2.circle(img, (int(pts1[x][0]), int(pts1[x][1])), 5, (0, 0, 255), cv2.FILLED)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
    display(Image.fromarray(img))

    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
    display(Image.fromarray(imgOutput))
    pasue()

    os.makedirs(output_dir, exist_ok=True)
    # 裁剪图像
    for index, file in tqdm(enumerate(input_data_dir), desc="Cropping", total=len(input_data_dir)):
        
        img = cv2.imread(file) # 打开图像
        cropped_image = cv2.warpPerspective(img,matrix,(width,height)) # 裁剪图像
        
        # 保存裁剪后的图像到指定目录
        cv2.imwrite(os.path.join(output_dir, 'frame_{:04d}.bmp'.format(index)), cropped_image)
        
    files_data = [os.path.join(output_dir, file) for file in os.listdir(output_dir)] # 列出文件夹下的文件路径    
    print("文件夹下的文件路径：", files_data[:3]) # 输出文件路径数组的前三个元素
    return files_data

def enhance_image(image):
    # 这一行创建了一个增强器对象，用于调整图像的对比度。
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(3.0)  # 以调整这个系数来增加或减少对比度。

    # 这一行将增强后的图像转换为灰度图像（'L'模式），即每个像素的强度用一个值表示（从黑到白）。
    gray_image = enhanced_image.convert('L')

    gray_array = np.array(gray_image)

    # 此行应用阈值处理将灰度图像转换为黑白图像。阈值为 128，低于此值的像素变为黑色（0），高于此值的像素变为白色（255）。cv2.THRESH_OTSU "是一种根据图像直方图自动确定阈值的方法。
    _, bw_image = cv2.threshold(gray_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 这一行将黑白 NumPy 数组转换回 PIL 图像，以便显示或进一步处理。
    final_image = Image.fromarray(bw_image)
    
    return final_image

def process_enhance(input_data_dir, output_dir):
    if if_check:
        image = Image.open(input_data_dir[0])
        display(image)
        final_image = enhance_image(image)
        display(final_image)
        pasue()
    os.makedirs(output_dir, exist_ok=True)
    # 处理所有图像
    for index, file in tqdm(enumerate(input_data_dir), desc="Enhancing", total=len(input_data_dir), unit="enhanced image"):
        image = Image.open(file)
        final_image = enhance_image(image)        
        
        # 保存裁剪后的图像到指定目录
        final_image.save(os.path.join(output_dir, 'frame_{:04d}.bmp'.format(index)))
    
    files_data = [os.path.join(output_dir, file) for file in os.listdir(output_dir)] # 列出文件夹下的文件路径    
    print("文件夹下的文件路径：", files_data[:3]) # 输出文件路径数组的前三个元素
    return files_data

# Parameters
if_pause = False
if_check = True
video_path = 'data/振动视频1.avi'
dir_temp = "data"
width,height=224,224

if __name__ == '__main__':
    dir_original_data = process_video(video_path, os.path.join(dir_temp, "original"))
    dir_cropped_data = process_cropped(dir_original_data, os.path.join(dir_temp, "cropped"), width, height)
    dir_processed_data = process_enhance(dir_cropped_data, os.path.join(dir_temp, "processed"))

def frame_processing():
    global if_pause, if_check
    if_pause, if_check = False, False
    print("Processing video...")
    dir_original_data = process_video(video_path, os.path.join(dir_temp, "original"))
    dir_cropped_data = process_cropped(dir_original_data, os.path.join(dir_temp, "cropped"), width, height)
    dir_processed_data = process_enhance(dir_cropped_data, os.path.join(dir_temp, "processed"))
    return dir_processed_data
