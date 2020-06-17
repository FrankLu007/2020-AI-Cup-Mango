import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ResNet101
from demo import WideRes
from argparser import get_args

def forward(name, dataloader, model, lossfunction = None, optimizer = None) :

	avgcorrect = [0.0] * 3
	cases = [0.0] * 3

	for i, (inputs, labels) in enumerate(dataloader):
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
		if lossfunction :
			loss = lossfunction(outputs, labels)
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
				avgcorrect[labels[i]] += 1
			cases[labels[i]] += 1
		del labels, pred

	# print result
	print('\t%s :'%name, '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
	for i in range(3) :
		print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))

	# return accuracy
	return sum(avgcorrect) / sum(cases)


if __name__ == '__main__' :

	args = get_args()
	ModelPath = 'C:\\Users\\Frank\\Machine Learning\\mango\\weight\\'
	DataPath = 'C:\\Users\\Frank\\Machine Learning\\mango\\mask\\'

	transform_train = transforms.Compose([
		transforms.RandomRotation(180),
		transforms.Resize((256, 256)),
		transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = transforms.Compose([
		transforms.Resize((512, 512)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	weights = torch.IntTensor([len(f) for r, d, f in os.walk(DataPath + "train\\")][1:])
	count = torch.Tensor([])
	total = sum(weights)
	for i in range(3) :
		count = torch.cat((count, torch.Tensor([total / float(weights[i]) for s in range(weights[i])])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = torchvision.datasets.ImageFolder(root = DataPath + 'train\\', transform = transform_train)
	validationset = torchvision.datasets.ImageFolder(root = DataPath + 'validation\\', transform = transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = 12, pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 32, shuffle = False, num_workers = args['thread'])

	if args['load'] :
		model = torch.load(ModelPath + args['load'])
	else :
		model = ResNet101(3).half().cuda()

	lossfunction = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	model.eval()
	with torch.no_grad() :
		bestaccuracy = forward('Validation', validationloader, model)

	if args['save'] :
		torch.save(model, ModelPath + args['save'])

	for epoch in range(args['epoch']) :

		print('\n\tEpoch : ' + str(epoch))

		model.train()
		forward('Training', trainloader, model, lossfunction, optimizer)

		model.eval()
		with torch.no_grad() :
			accuracy = forward('Validation', validationloader, model)

		if args['save'] and accuracy > bestaccuracy:
			bestaccuracy = accuracy
			torch.save(model, ModelPath + args['save'])