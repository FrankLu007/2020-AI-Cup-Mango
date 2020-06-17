import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets.folder import pil_loader
import random

transform_cut = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
recover = transforms.ToPILImage()
Tensor = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True).cuda()
model.eval()

def cut(image) :
    image = transform_cut(image).reshape(1, 3, 512, 512).cuda()
    outputs = model(image)
    del image
    A = 0
    B = 0
    C = 0
    D = 0
    if 3 not in outputs[0]['labels'] and 1 not in outputs[0]['labels'] and 67 not in outputs[0]['labels'] and 64 not in outputs[0]['labels']:
        return 0, 0, 0, 0

    for index in range(len(outputs[0]['boxes'])):
        if outputs[0]['labels'][index] != 3 and outputs[0]['labels'][index] != 1 and outputs[0]['labels'][index] != 67 and outputs[0]['labels'][index] != 64:
            continue
        a = int(outputs[0]['boxes'][index][0])
        b = int(outputs[0]['boxes'][index][2])
        c = int(outputs[0]['boxes'][index][1])
        d = int(outputs[0]['boxes'][index][3])
        if (b - a) * (d - c) > (B - A) * (D - C):
            A, B, C, D = a, b, c, d

    return A, B, C, D

for dirPath, dirNames, fileNames in os.walk('C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\test'):
    if dirNames == []:
        for files in fileNames:
            image = pil_loader(dirPath + '/' + files)
            a, b, c, d = cut(image)
            # print(a, b, c, d)
            # plt.figure(figsize = (50, 10))
            # plt.imshow(image)
            if (b - a) * (d - c) == 0:
                X = image
            else :    
                X = recover(Tensor(image)[:, a:b, c:d])
            X.save(dirPath.replace('dataset', 'mask') + '/' + files)
            # plt.figure(figsize = (50, 10))
            # plt.imshow(X)