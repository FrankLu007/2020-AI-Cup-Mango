import torch.nn
import torchvision.models
import torch.utils.model_zoo
from efficientnet_pytorch import EfficientNet
# from vit_pytorch import ViT

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b6')
        self.fc = torch.nn.Linear(1000, 3)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

# class Vision_Transformer(torch.nn.Module):
#     def __init__(self):
#         super(Vision_Transformer, self).__init__()
#         self.ImageNet = ViT(image_size = 224, patch_size = 32, num_classes = 3, dim = 1280, depth = 32, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)
#     def forward(self, x):
#         return self.ImageNet(x)

def resnext101_32x48d_wsl(progress = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    model = torchvision.models.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
    state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth', progress = progress)
    model.load_state_dict(state_dict)
    return model

class FixResNet(torch.nn.Module):
    def __init__(self):
        super(FixResNet, self).__init__()
        self.ImageNet = resnext101_32x48d_wsl()
        self.fc = torch.nn.Linear(1000, 3)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

