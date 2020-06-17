import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader

class WideRes(nn.Module):
    def __init__(self):
        super(WideRes, self).__init__()
        self.ImageNet = torchvision.models.wide_resnet101_2(pretrained = True)
        self.fc1 = nn.Linear(1000, 64)
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.ImageNet(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def forward(image, model):

    inputs = image.half().cuda()
    outputs = model(inputs)
    del image, inputs

    res = outputs.sum(0)
    tmp, pred = res.max(0)
    del tmp, outputs

    return pred

Norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
def split(images):

    inputs = torch.stack([Norm(image) for image in images])
    return inputs

if __name__ == '__main__' :

    DataPath = 'C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\test\\'
    ModelPath = 'C:\\Users\\Frank\\Machine Learning\\mango\\weight\\'

    # transform_demo = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])

    transform_demo = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.FiveCrop((224, 224)),
        transforms.Lambda(split),
    ])

    model = torch.load(ModelPath + sys.argv[1])
    model.eval()

    print('image_id,label')

    result = ['A', 'B', 'C']
    for _, _, files in os.walk(DataPath):
        for file in files :
            image = transform_demo(pil_loader(DataPath + file)).reshape(-1, 3, 224, 224)
            print(file + ',' + result[forward(image, model)])
