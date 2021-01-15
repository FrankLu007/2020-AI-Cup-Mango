import os
import csv
import torch, numpy
import torchvision.transforms
import torchvision.datasets
from PIL import Image
from model import EfficientNetWithFC
from argparser import get_args

def LoadCSV(file):
    with open(file, 'r', encoding = 'UTF-8') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        return list(csv_reader)

def cut(image, a, b, c, d):
    s = numpy.array(image)
    z = numpy.zeros(s.shape, dtype = s.dtype)
    z[b:d, a:c] = s[b:d, a:c]
    return Image.fromarray(z)

if __name__ == '__main__' :

    args = get_args()
    ModelPath = '../../'
    DataPath = '../../Test/'

    transform_demo = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args['size'], args['size'])),
        torchvision.transforms.RandomRotation(180),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.GaussianBlur(7, 10),
        torchvision.transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    DataList = LoadCSV('../../Test_mangoXYWH.csv')
    model = torch.load(ModelPath + args['load'])
    model.eval()
    S = torch.nn.Sigmoid()
    torch.backends.cudnn.benchmark = True

    print('image_id,D1,D2,D3,D4,D5')

    index = 5377
    while index < len(DataList): 
        time = args['ep']
        result = torch.zeros(len(DataList[index : index + args['bs']]), 5).cuda()
#         ImageList = [cut(torchvision.datasets.folder.pil_loader(DataPath + file[0]), int(file[1]), int(file[2]), int(file[1]) + int(file[3]), int(file[2]) + int(file[4])) for file in DataList[index : index + args['bs']]]
        ImageList = [torchvision.datasets.folder.pil_loader(DataPath + file[0]) for file in DataList[index : index + args['bs']]]
        while time :
            images = torch.stack([transform_demo(image) for image in ImageList]).reshape(-1, 3, args['size'], args['size']).cuda()
            with torch.no_grad() :
                with torch.cuda.amp.autocast():
                    result += S(model(images).detach())
            del images
            time -= 1
        del ImageList
#         result[:, 0] *= 0.6
#         result[:, 1] *= 0.6
#         result[:, 2] *= 0.7
#         result[:, 3] *= 1.8
#         result[:, 4] *= 0.6
        result[result < args['ep'] * 0.5] = 0
        result[result >= args['ep'] * 0.5] = 1
        for i, file in enumerate(DataList[index : index + args['bs']]):
            print(file[0] + ',' + str(int(result[i][0].item())) + ',' + str(int(result[i][1].item())) + ',' + str(int(result[i][2].item())) + ',' + str(int(result[i][3].item())) + ',' + str(int(result[i][4].item())))
        index += args['bs']
        del result
