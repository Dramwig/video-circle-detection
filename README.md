# 视频帧小圈的目标检测与定位

使用 PyTorch 定位视频帧中的圆环

## 简介

本资源库包含用于视频帧提取、图像处理、数据集生成和模型训练的脚本和笔记本，重点关注图像分析任务。这些组件旨在支持信号处理、计算机视觉和机器学习等领域的研究与开发。

## 项目结构

以下是项目目录的概览：

```
D:.
│  output_values.xlsx
│  README.md
│  resnet50_regression_0.0310.pth
│  数据处理和生成.ipynb
│  模型和训练.ipynb
├─data
│  │  振动视频1.avi
│  ├─cropped 
│  ├─original  
│  ├─processed   
│  ├─test   
│  ├─train    
│  └─validation        
└─img
```

### 目录说明

- **data**： 包含项目中使用的所有数据集和视频文件。
 - **cropped**： 存储从视频中裁剪的图像。
 - **original**： 包含从视频中提取的原始帧。
 - **processed**： 保存经过处理的图像，以备模型输入。
 - **train/test/validation**： 包含用于训练、测试和验证模型的图像和中心坐标文件。

## 功能与目的

1. **视频帧提取**： 脚本首先从视频文件中提取帧，并将其保存为 BMP 图像，为进一步处理做好准备。

2. **图像裁剪和透视变换**： 它通过裁剪和应用透视变换来处理图像，以聚焦于帧内的特定区域。

3. **图像增强和二值化**： 该部分可增强图像的对比度并将其转换为二值图像（黑白），这对于某些不需要彩色数据的分析类型非常有用。

4. **生成合成图像和中心**： 该脚本的特色是通过绘制圆环和添加噪点和污点来模拟现实世界中的瑕疵，从而生成合成图像。这部分对于创建一个接近实际场景的数据集至关重要。

5. **数据集组织与处理**： 它将生成的图像及其相应的元数据（如某些特征的中心坐标）组织成训练集、验证集和测试集。这对于机器学习模型的训练至关重要。

6. **模型训练**： 利用 PyTorch 来训练修改后的 ResNet50 模型，该模型适用于回归任务，专为处理单通道（灰度）图像而设计。

7. **评估和输出**： 训练完成后，使用测试集对该模型进行评估，并将结果可视化和保存。这有助于了解模型的性能。

8. **输出值的傅立叶变换**： 该脚本可能打算使用快速傅里叶变换分析输出的频率成分，这在振动分析等应用或任何与频率分析相关的场景中都很有用。

该脚本适用于需要进行全面预处理、分析和模型训练的学术和研究用途。它在机器学习、计算机视觉和数字信号处理等领域尤其有用。

## 设置

### 先决条件

确保已安装 Python 3.x 和以下软件包：
- OpenCV
- NumPy
- PyTorch 和 torchvision
- PIL (Pillow)
- matplotlib
- tqdm

#### 安装

克隆软件源并导航至目录：
```bash
git clone <repository-url>
cd <仓库目录>
```
安装所需的 Python 软件包：

```
while read requirement; do conda install --yes $requirement; done < requirements.txt
# 或者
pip install -r requirements.txt
```

## 使用方法

要运行脚本，请导航到项目的根目录，然后直接执行笔记本或 Python 脚本。例如，处理图像：

```bash
python 数据处理和生成.ipynb
```
并训练模型：
```bash
python 模型和训练.ipynb
```

## 投稿

欢迎贡献。请 fork 代码库，并提交拉取请求。

## 许可证

本项目采用 MIT 许可，详情请参见 LICENSE.md 文件。
