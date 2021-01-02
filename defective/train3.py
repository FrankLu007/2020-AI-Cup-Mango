import os, csv
import torch
import torchvision.transforms
import torchvision.datasets
from PIL import Image
from model import ResNet
from argparser import get_args

def LoadCSV(file):
	with open(file, 'r', encoding = 'UTF-8') as file:
		csv_reader = csv.reader(file, delimiter = ',')
		data = list(csv_reader)
	for d in data:
		for index in range(1, len(d)):
			d[index] = int(d[index])
	return data

class MangoDataset(torch.utils.data.Dataset):
	def __init__(self, path, file, transform, training = False):
		super(MangoDataset, self).__init__()
		self.path = path
		self.file = LoadCSV(file)
		self.transform = transform
		self.mode = training
		self.Beta = torch.distributions.beta.Beta(0.5, 0.5)
	def __len__(self):
		return len(self.file)
	def __getitem__(self, index):
		image = None
		if self.mode and torch.rand(1) < 0.5:
			beta = self.Beta.sample().item()
			partner = int(self.__len__() * torch.rand(1))
			A = Image.open(self.path + self.file[index][0])
			B = Image.open(self.path + self.file[partner][0])
			label = torch.Tensor(self.file[index][1:]) * beta + (1 - beta) * torch.Tensor(self.file[partner][1:])
			image = self.transform(A) * beta + self.transform(B) * (1 - beta)
			del beta, partner, A, B
			return image, label
		else :
			image = Image.open(self.path + self.file[index][0])
			label = torch.Tensor(self.file[index][1:])
			return self.transform(image), label
		

def forward(DataLoader, model, LossFunction, optimizer = None, scaler = None) :

	precision = [0.0] * 5
	recall = [0.0] * 5
	correct = [0.0] * 5
	predict = [0.0] * 5
	cases = [0.0] * 5
	TotalLoss = 0.0

	for _, (inputs, labels) in enumerate(DataLoader):
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
		# forward
		inputs = inputs.half().cuda()
		with torch.cuda.amp.autocast():
			outputs = model(inputs)
			del inputs

			# loss and step
			labels = labels.cuda()
			loss = LossFunction(outputs, labels)
			TotalLoss += loss.item()

		if optimizer :
			# loss.backward()
			scaler.scale(loss).backward()
		del loss
		if optimizer :
			scaler.step(optimizer)
			scaler.update()
		
		if optimizer == None:
			
			pred = outputs > 0.5

			# calculate accuracy
			for c in range(5):
				cases[c] += (labels[:, c] == 1).sum().item()
				predict[c] += pred[:, c].sum().item()
				correct[c] += (labels[:, c] * pred[:, c]).sum().item()
			del pred

		del outputs

	if optimizer == None:
		# print result
		for index in range(5):
			if predict[index] :
				precision[index] = correct[index] / predict[index] * 100
			else :
				precision[index] = 0
			recall[index] = correct[index] / cases[index] * 100
		P = sum(precision) / 5
		R = sum(recall) / 5

		print('%5.2f%%'%(2 * R * P / (R + P)), TotalLoss / len(DataLoader))
		for c in range(5):
			if precision[c] + recall[c]:
				print('\tclass %d'%c, '%6d'%cases[c], '%5.2f%%'%precision[c], '%5.2f%%'%recall[c], '%5.2f%%'%(2 * precision[c] * recall[c] / (precision[c] + recall[c])))
			else :
				print('\tclass %d'%c, '%6d'%cases[c], '%5.2f%%'%precision[c], '%5.2f%%'%recall[c], '0.00%%')

		# return accuracy
		del correct, cases, TotalLoss, precision, recall
		return 2 * R * P / (R + P)
	else :
		return TotalLoss


if __name__ == '__main__' :

	args = get_args()
	ModelPath = '../../'
	DataPath = '..\\..\\'

	transform_train = torchvision.transforms.Compose([
		torchvision.transforms.RandomRotation(180),
		torchvision.transforms.Resize((args['size'], args['size'])),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.GaussianBlur(7, 10),
		torchvision.transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = torchvision.transforms.Compose([
		torchvision.transforms.Resize((args['size'], args['size'])),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	TrainingSet = MangoDataset(path = DataPath, file = '..\\..\\training.csv', transform = transform_train, training = True)
	ValidationSet = MangoDataset(path = DataPath, file = '..\\..\\validation.csv', transform = transform_test)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args['bs'], num_workers = 2, pin_memory = True, drop_last = True, shuffle = True, prefetch_factor = 1)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = args['bs'], num_workers = 2, prefetch_factor = 1)

	if args['load'] :
		model = torch.load(ModelPath + args['load'])
	else :
 		model = ResNet().cuda()
	torch.backends.cudnn.benchmark = True
	LossFunction = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'], momentum = 0.9, weight_decay = args['lr'] * 0.001)
	scaler = torch.cuda.amp.GradScaler()

	model.eval()
	with torch.no_grad() :
		print('Validatoin', end = ' ')
		bestaccuracy = forward(ValidationLoader, model, LossFunction)

	if args['save'] :
		torch.save(model, ModelPath + args['save'])

	for epoch in range(args['ep']) :

		print('\nEpoch : ' + str(epoch))

		model.train()
		print('Training', end = ' ')
		forward(TrainingLoader, model, LossFunction, optimizer, scaler)

		model.eval()
		with torch.no_grad() :
			print('Validatoin', end = ' ')
			accuracy = forward(ValidationLoader, model, LossFunction)

		if args['save'] and accuracy > bestaccuracy:
			bestaccuracy = accuracy
			torch.save(model, ModelPath + args['save'])


