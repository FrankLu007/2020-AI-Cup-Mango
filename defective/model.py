import torch.nn
import torchvision.models
import torch.utils.model_zoo
from efficientnet_pytorch import EfficientNet

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.ImageNet = torchvision.models.resnet152(pretrained = True, progress = True)
        self.fc = torch.nn.Linear(1000, 5)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

class EfficientNetWithFC(torch.nn.Module):
    def __init__(self):
        super(EfficientNetWithFC, self).__init__()
        self.ImageNet = EfficientNet.from_pretrained('efficientnet-b7')
        self.fc = torch.nn.Linear(1000, 5)
    def forward(self, x):
        return self.fc(self.ImageNet(x))

class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        for m in models:
            m.eval()
        self.fc1 = torch.nn.Linear(2000, 2000)
        self.fc2 = torch.nn.Linear(2000, 2000)
        self.fc3 = torch.nn.Linear(2000, 5)
        self.relu = torch.nn.ReLU(inplace = True)
    def forward(self, x):
        with torch.no_grad():
            x = torch.cat(tuple(m.ImageNet(x) for m in self.models), dim = 1)
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

class Boosting(torch.nn.Module):
    def __init__(self, models):
        super(Boosting, self).__init__()
        self.models = models
        for m in models:
            m.eval()
        self.fc1 = torch.nn.Linear(2560 * 256 + 2304 * 256, 256)
        self.fc2 = torch.nn.Linear(256, 5)
        self.relu = torch.nn.ReLU(inplace = True)
    def forward(self, x):
        bs = x.shape[0]
        with torch.no_grad():
            x = torch.cat(tuple(m.ImageNet.extract_features(x).reshape(bs, -1) for m in self.models), dim = 1)
        return self.fc2(self.relu(self.fc1(x)))


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

