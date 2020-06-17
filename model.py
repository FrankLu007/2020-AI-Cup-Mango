import torch
import torch.nn as nn
import torchvision

class ResNet101(nn.Module):
    def __init__(self, output_size):
        super(ResNet101, self).__init__()
        self.ImageNet = torchvision.models.resnext101_32x8d(pretrained = True)
        self.fc = nn.Linear(1000, output_size)
        
    def forward(self, x):
        x = self.ImageNet(x)
        return self.fc(x)

