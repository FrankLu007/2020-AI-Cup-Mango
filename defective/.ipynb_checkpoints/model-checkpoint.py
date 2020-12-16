import torch.nn
import torchvision.models
import torch.utils.model_zoo
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_name('efficientnet-l2')
        self.fc = torch.nn.Linear(1000, 5)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        for m in models:
            m.eval()
        self.fc1 = torch.nn.Linear(2005, 2005)
        self.fc2 = torch.nn.Linear(2005, 5)
        self.relu = torch.nn.ReLU(inplace = True)
    def forward(self, x):
        with torch.no_grad():
            x = torch.cat(tuple(m.ImageNet(x) for m in self.models), dim = 1)
        return self.fc2(self.relu(self.fc1(x)))


class Vision_Transformer(torch.nn.Module):
    def __init__(self):
        super(Vision_Transformer, self).__init__()
        self.ImageNet = ViT(image_size = 512, patch_size = 32, num_classes = 5, dim = 256, depth = 24, heads = 16, mlp_dim = 256, dropout = 0.1, emb_dropout = 0.1)
    def forward(self, x):
        return self.ImageNet(x)

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
        self.fc = torch.nn.Linear(1000, 5)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

