import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def show_dataset(dataset, n = 5):
	img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(5)))
	plt.imshow(img)
	plt.axis('off')

transform_train = transforms.Compose([
	transforms.Resize((512, 512)),
	torchvision.transforms.GaussianBlur(7, 10),
	transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025), #b = (0.75, 1.25), c = (0.5, 2), s = (1, 1.5), h = 0.05
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == '__main__' :
	show_dataset(torchvision.datasets.ImageFolder(root = '../MangoData/train/', transform = transform_train))
	plt.show()