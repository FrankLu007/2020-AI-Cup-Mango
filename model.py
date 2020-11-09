import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b6')
        self.fc = nn.Linear(1000, 3)
        
    def forward(self, x):
        x = self.ImageNet(x)
        return self.fc(x)
    
class ViTNet(nn.Module):
    def __init__(self):
        super(ViTNet, self).__init__()
        self.ImageNet = ViT(image_size = 256, patch_size = 16, num_classes = 3, dim = 256, depth = 32, heads = 16, mlp_dim = 256, dropout = 0.1, emb_dropout = 0.1)
        
    def forward(self, x):
        return self.ImageNet(x)

