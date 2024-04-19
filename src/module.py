import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from src.config import default_model_type

# Modify ResNet50
class ResNet50Regression(nn.Module):
    def __init__(self):
        super(ResNet50Regression, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the first convolutional layer to accept single-channel input
        self.resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-1]
        )
        
        # Replace the last fully connected layer to output a single value
        self.regressor = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
    
# ----------------------

# Define the Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modify the ResNet Block to include the SE Block
class SEResNetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# Modify ResNet50 to SE-ResNet50 with single-channel input and regression output
class SEResNet50Regression(nn.Module):
    def __init__(self):
        super(SEResNet50Regression, self).__init__()
        # resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.inplanes = 64
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(SEResNetBottleneck, 64, 3)
        self.layer2 = self._make_layer(SEResNetBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(SEResNetBottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(SEResNetBottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(512 * SEResNetBottleneck.expansion, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # 如果输入输出通道数不同
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x
    
# ----------------------
                            
class UserModule(nn.Module):
    def __init__(self, model_type=default_model_type):
        super(UserModule, self).__init__()
        if model_type == 'resnet':
            self.model = ResNet50Regression()
            self.model_type = 'resnet'
        elif model_type == 'seresnet':
            self.model = SEResNet50Regression()
            self.model_type = 'seresnet'
        else:
            raise ValueError("Unsupported model type. Please choose between 'resnet' and 'seresnet'.")

    def forward(self, x):
        return self.model(x)
