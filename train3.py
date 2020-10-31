import os
import torch
import torchvision
import torchvision.transforms as transforms
from model import ImageNet
from argparser import get_args

def forward(DataLoader, model, LossFunction, optimizer = None) :

	correct = [0.0] * 3
	cases = [0.0] * 3
	TotalLoss = 0.0

	for _, (inputs, labels) in enumerate(DataLoader):
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
		# forward
		inputs = inputs.half().cuda()
		outputs = model(inputs)
		del inputs

		# loss and step
		labels = labels.cuda()
		loss = LossFunction(outputs, labels)
		TotalLoss += loss.item()
		if optimizer :
			loss.backward()
		del loss
		if optimizer :
			optimizer.step()
		
		# convert to prediction
		tmp, pred = outputs.max(1)
		del tmp, outputs
		
		# calculate accuracy
		for i in range(len(pred)) :
			if pred[i] == labels[i] :
				correct[labels[i]] += 1
			cases[labels[i]] += 1
		del labels, pred

	# print result
	print('%.2f%%'%(sum(correct) / sum(cases) * 100), '%5.2f'%(TotalLoss/len(DataLoader)))
	for i in range(3) :
		print('\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(correct[i] / cases[i] * 100))

	# return accuracy
	accuracy = sum(correct) / sum(cases)
	del correct, cases, TotalLoss
	return accuracy


if __name__ == '__main__' :

	args = get_args()
	ModelPath = '..\\'
	DataPath = '..\\MangoData\\'

	transform_train = transforms.Compose([
		transforms.RandomRotation(180),
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = transforms.Compose([
		transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	weights = torch.IntTensor([len(f) for r, d, f in os.walk(DataPath + "train\\")][1:])
	total = int(sum(weights))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor([total / float(weights[0]) for _ in range(weights[0])] + [total / float(weights[1]) for _ in range(weights[1])] + [total / float(weights[2]) for _ in range(weights[2])]), total)
	del weights

	TrainingSet = torchvision.datasets.ImageFolder(root = DataPath + 'train\\', transform = transform_train)
	ValidationSet = torchvision.datasets.ImageFolder(root = DataPath + 'validation\\', transform = transform_test)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args['bs'], num_workers = 8, pin_memory = True, drop_last = True, sampler = sampler)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = 128, num_workers = 12)

	if args['load'] :
		model = torch.load(ModelPath + args['load'])
	else :
		model = ImageNet().half().cuda()

	LossFunction = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'], momentum = 0.9)

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
		forward(TrainingLoader, model, LossFunction, optimizer)

		model.eval()
		with torch.no_grad() :
			print('Validatoin', end = ' ')
			accuracy = forward(ValidationLoader, model, LossFunction)

		if args['save'] and accuracy > bestaccuracy:
			bestaccuracy = accuracy
			torch.save(model, ModelPath + args['save'])