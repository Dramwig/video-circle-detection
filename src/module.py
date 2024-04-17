import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

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