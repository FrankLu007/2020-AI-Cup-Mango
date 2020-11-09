import torch.nn
import torchvision.models
import torch.utils.model_zoo
from efficientnet_pytorch import EfficientNet

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b6')
        self.fc = torch.nn.Linear(1000, 3)
        
    def forward(self, x):
        x = self.ImageNet(x)
        return self.fc(x)

def resnext101_32x48d_wsl(progress = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    model = torchvision.models.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], **kwargs)
    state_dict = torch.utils.model_zoo.load_url('https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth', progress = progress)
    model.load_state_dict(state_dict)
    return model


