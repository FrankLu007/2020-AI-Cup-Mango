import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
from model import ResNet101
from argparser import get_args

def forward(name, dataloader, model, target_class, lossfunction = None, optimizer = None) :

	totalloss = 0.0
	iteration = 0
	avgcorrect = [[0.0, 0.0], [0.0, 0.0]]

	for i, (inputs, labels) in enumerate(dataloader):
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
		for index in range(len(labels)) :
			if labels[index] != target_class :
				labels[index] = 0
			else :
				labels[index] = 1

		# forward
		inputs = inputs.half().cuda()
		outputs = model(inputs)
		del inputs

		# loss and step
		labels = labels.half().cuda()
		loss = lossfunction(outputs.reshape(-1), labels)
		totalloss += loss.item()

		if optimizer :
			loss.backward()
			optimizer.step()
		del loss
		
		# calculate accuracy
		for i in range(len(labels)) :
			ans = int(labels[i])
			if outputs[i] >= 0.5 and ans == 1:
				avgcorrect[0][0] += 1
			elif outputs[i] < 0.5 and ans == 0:
				avgcorrect[1][1] += 1
			elif outputs[i] >= 0.5 and ans == 0:
				avgcorrect[1][0] += 1
			elif outputs[i] < 0.5 and ans == 1:
				avgcorrect[0][1] += 1
			else:
				print('WTF')
		iteration += 1
		del labels, outputs, ans

	# print result
	if avgcorrect[0][0] + avgcorrect[1][0] == 0:
		Precision = 0
		print('WTF: P = 0', avgcorrect[0][0], avgcorrect[1][0])
	else :
		Precision = avgcorrect[0][0] / (avgcorrect[0][0] + avgcorrect[1][0])

	if avgcorrect[0][0] + avgcorrect[0][1] == 0:
		Recall = 0
		print('WTF: R = 0', avgcorrect[0][0], avgcorrect[0][1])		
	else :
		Recall = avgcorrect[0][0] / (avgcorrect[0][0] + avgcorrect[0][1])
	print('\t%s :'%name, 'Loss: %5.3f'%(totalloss / iteration), 'Precision: %5.2f'%(Precision), 'Recall: %5.2f'%(Recall))

	# return loss
	return totalloss


if __name__ == '__main__' :

	args = get_args()

	transform_train = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.RandomHorizontalFlip(),
	    transforms.ColorJitter(0.5, 0.5, 0.5),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	weights = []
	target_no = -1
	for root, directory, files in os.walk("C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\train\\") :
		if root == "C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\train\\B":
			target_no = len(weights)
		if len(directory) == 0 :
			weights += [len(files)]
	count = torch.Tensor([])
	total = sum(weights)
	total_not_target = total - weights[target_no]
	for i in range(3) :
		count = torch.cat((count, torch.Tensor([total / (weights[i] if i == target_no else total_not_target) for s in range(weights[i])])))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(count, len(count))

	trainset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\train\\', transform = transform_train)
	validationset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\validation\\', transform = transform_test)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = args['batch_size'], shuffle = False, num_workers = args['thread'], pin_memory = True, drop_last = True, sampler = sampler)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 128, shuffle = False, num_workers = args['thread'])

	if args['load'] :
		model = torch.load('C:\\Users\\Frank\\Machine Learning\\mango\\weight\\' + args['load'])
	else :
		model = ResNet101(1).half().cuda()

	lossfunction = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr = args['learning_rate'], momentum = 0.9)

	print('Training :')

	model.eval()
	with torch.no_grad() :
		bestlost = forward('Validation', validationloader, model, args['class'], lossfunction)

	if args['save'] :
		torch.save(model, 'C:\\Users\\Frank\\Machine Learning\\mango\\weight\\' + args['save'])

	for epoch in range(args['epoch']) :

		print('\n\tEpoch : ' + str(epoch))

		model.train()
		forward('Training', trainloader, model, args['class'], lossfunction, optimizer)

		model.eval()
		with torch.no_grad() :
			lost = forward('Validation', validationloader, model, args['class'], lossfunction)

		if args['save'] and lost < bestlost:
			bestlost = lost
			torch.save(model, 'C:\\Users\\Frank\\Machine Learning\\mango\\weight\\' + args['save'])