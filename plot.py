import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def show_dataset(dataset, n = 5):
	img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n))) for i in range(10)))
	plt.imshow(img)
	plt.axis('off')

transform_train = transforms.Compose([
	transforms.RandomRotation(90),
	transforms.Resize((256, 256)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomGrayscale(1),
	transforms.ColorJitter(contrast = (2, 10)),
    # transforms.ToTensor(),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == '__main__' :
	show_dataset(torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\', transform = transform_train))
	plt.show()