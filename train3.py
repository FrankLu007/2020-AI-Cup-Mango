import os
import torch
import torchvision.transforms
import torchvision.datasets
from model import EfficientNetWithFC, resnext101_32x48d_wsl
from argparser import get_args

def forward(DataLoader, model, LossFunction, optimizer = None, scaler = None) :

	correct = [0.0] * 3
	cases = [0.0] * 3
	TotalLoss = 0.0

	for _, (inputs, labels) in enumerate(DataLoader):
		# initialize
		torch.cuda.empty_cache()
		if optimizer :
			optimizer.zero_grad()
		# forward
		inputs = inputs.cuda()
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
	print('%.2f%%'%(sum(correct) / sum(cases) * 100),'/ %.2f%%'%(sum([correct[i] / cases[i] * 100 / 3.0 for i in range(len(cases))])), '%5.3f'%(TotalLoss/len(DataLoader)))
	for i in range(3) :
		print('\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(correct[i] / cases[i] * 100))

	# return accuracy
	accuracy = sum(correct) / sum(cases)
	del correct, cases, TotalLoss
	return accuracy


if __name__ == '__main__' :

	args = get_args()
	ModelPath = '../'
	DataPath = '../MangoData/merge/'

	transform_train = torchvision.transforms.Compose([
		torchvision.transforms.RandomRotation(180),
		torchvision.transforms.Resize((512, 512)),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.GaussianBlur(7, 10),
		torchvision.transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	transform_test = torchvision.transforms.Compose([
		torchvision.transforms.Resize((512, 512)),
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	weights = torch.IntTensor([len(f) for _, _, f in os.walk(DataPath + 'train/')])[1:]
	total = int(sum(weights))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor([total / float(weights[0]) for _ in range(weights[0])] + [total / float(weights[1]) for _ in range(weights[1])] + [total / float(weights[2]) for _ in range(weights[2])]), total)
	del weights

	TrainingSet = torchvision.datasets.ImageFolder(root = DataPath + 'train/', transform = transform_train)
	ValidationSet = torchvision.datasets.ImageFolder(root = DataPath + 'validation/', transform = transform_test)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args['bs'], num_workers = 8, pin_memory = True, drop_last = True, sampler = sampler)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = 128, num_workers = 8)

	if args['load'] :
		model = torch.load(ModelPath + args['load'])
	else :
		model = resnext101_32x48d_wsl().cuda()
		# model = EfficientNetWithFC().cuda()

	LossFunction = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'], momentum = 0.9)
	scaler = torch.cuda.amp.GradScaler()
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 3, verbose = True)

	model.eval()
	with torch.no_grad() :
		print('Validatoin', end = ' ')
		bestaccuracy = forward(ValidationLoader, model, LossFunction)
		# scheduler.step(bestaccuracy)

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
			# scheduler.step(accuracy)

		if args['save'] and accuracy > bestaccuracy:
			bestaccuracy = accuracy
			torch.save(model, ModelPath + args['save'])

	del TrainingSet, ValidationSet, TrainingLoader, ValidationLoader, sampler, optimizer
	TestingSet = torchvision.datasets.ImageFolder(root = DataPath + 'test/', transform = transform_train)
	TestingLoader = torch.utils.data.DataLoader(TestingSet, batch_size = 128, num_workers = 8)

	result = torch.zeros((len(TestingSet), 3))

	if args['save'] :
		del model
		model = torch.load(ModelPath + args['save'])
	with torch.no_grad() :
		for _ in range(2):
			for index, (inputs, labels) in enumerate(TestingLoader):
				torch.cuda.empty_cache()
				inputs = inputs.cuda()
				with torch.cuda.amp.autocast():
					outputs = model(inputs)
				result[index * 128 : index * 128 + 128] += outputs.detach().cpu()
				del inputs, outputs, labels
		cases = [0.0, 0.0, 0.0]
		correct = [0.0, 0.0, 0.0]
		for index, (inputs, labels) in enumerate(TestingLoader):
			torch.cuda.empty_cache()
			inputs = inputs.cuda()
			with torch.cuda.amp.autocast():
				outputs = model(inputs)
			result[index * 128 : index * 128 + 128] += outputs.detach().cpu()
			del inputs, outputs
			tmp, pred = result[index * 128 : index * 128 + 128].max(1)
			
			# calculate accuracy
			for i in range(len(pred)) :
				if pred[i] == labels[i] :
					correct[labels[i]] += 1
				cases[labels[i]] += 1
			del tmp

	print('Test', '%.2f%%'%(sum(correct) / sum(cases) * 100),'/ %.2f%%'%(sum([correct[i] / cases[i] * 100 / 3.0 for i in range(len(cases))])))
	for i in range(3) :
		print('\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(correct[i] / cases[i] * 100))


