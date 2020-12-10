import os
import csv
import torch
import torchvision.transforms
import torchvision.datasets
from model import EfficientNetWithFC
from argparser import get_args

def LoadCSV(file):
    with open(file, 'r') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        return list(csv_reader)[1:]

if __name__ == '__main__' :

    args = get_args()
    ModelPath = '../'
    DataPath = '../MangoData/merge/test/'

    transform_demo = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args['size'], args['size'])),
        torchvision.transforms.RandomRotation(180),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.GaussianBlur(7, 10),
        torchvision.transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    DataList = LoadCSV('../test_Final_example.csv')
    model = torch.load(ModelPath + args['load'])
    model.eval()

    print('image_id,label')

    label = ['A', 'B', 'C']
    index = 0
    while index < len(DataList): 
        time = args['ep']
        result = torch.zeros(len(DataList[index : index + args['bs']]), 3).cuda()
        ImageList = [torchvision.datasets.folder.pil_loader(DataPath + file[0]) for file in DataList[index : index + args['bs']]]
        while time :
            images = torch.stack([transform_demo(image) for image in ImageList]).reshape(-1, 3, args['size'], args['size']).cuda()
            with torch.no_grad() :
                with torch.cuda.amp.autocast():
                    result += model(images).detach()
            del images
            time -= 1
        del ImageList
        tmp, pred = result.max(1)
        for i, file in enumerate(DataList[index : index + args['bs']]):
            print(file[0] + ',' + label[pred[i]])
        index += args['bs']
        del tmp, pred, result
