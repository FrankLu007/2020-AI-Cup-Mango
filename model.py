import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b6')
        self.fc = nn.Linear(1000, 3)
        
    def forward(self, x):
        x = self.ImageNet(x)
        return self.fc(x)

