import os
import torch
import torchvision.transforms
import torchvision.datasets
from model import EfficientNetWithFC, Vision_Transformer, FixResNet
from argparser import get_args

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
		
		# convert to prediction
		tmp, pred = outputs.detach().max(1)
		del tmp, outputs
		
		# calculate accuracy
		for i in range(len(pred)) :
			if pred[i] == labels[i] :
				correct[labels[i]] += 1
			predict[pred[i]] += 1
			cases[labels[i]] += 1
		del labels, pred

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


if __name__ == '__main__' :

	args = get_args()
	ModelPath = '../'
	DataPath = '../../data/'

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

	weights = torch.IntTensor([2705, 532, 24447, 15373, 1779])
	total = int(sum(weights))
	sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor([total / float(weights[0]) for _ in range(weights[0])] + [total / float(weights[1]) for _ in range(weights[1])] + [total / float(weights[2]) for _ in range(weights[2])] + [total / float(weights[3]) for _ in range(weights[3])] + [total / float(weights[4]) for _ in range(weights[4])]), total)
	del weights

	TrainingSet = torchvision.datasets.ImageFolder(root = DataPath + 'train/', transform = transform_train)
	ValidationSet = torchvision.datasets.ImageFolder(root = DataPath + 'validation/', transform = transform_test)

	TrainingLoader = torch.utils.data.DataLoader(TrainingSet, batch_size = args['bs'], num_workers = 32, pin_memory = True, drop_last = True, sampler = sampler)
	ValidationLoader = torch.utils.data.DataLoader(ValidationSet, batch_size = args['bs'], num_workers = 32)

	if args['load'] :
		model = torch.load(ModelPath + args['load'])
	else :
		model = FixResNet().cuda()
# 		model = Vision_Transformer().cuda()
#  		model = EfficientNetWithFC().cuda()
	torch.backends.cudnn.benchmark = True
	LossFunction = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'], momentum = 0.9)
	scaler = torch.cuda.amp.GradScaler()

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


