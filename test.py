import os
import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet101

def single_model(inputs) :
	model = torch.load('C:\\Users\\Frank\\Machine Learning\\mango\\weight\\ResNet101_1_1_Flip_Color')
	outputs = model(inputs)
	# pred = outputs >= 0.8
	tmp, pred = outputs.max(1)
	del model, outputs, tmp

	return pred

def overall_model(inputs) :
	model = torch.load('C:\\Users\\Frank\\Machine Learning\\mango\\weight\\ResNet101_3_Flip_Color')
	outputs = model(inputs)
	tmp, pred = outputs.max(1)
	del tmp, model, outputs

	return pred

def forward(dataloader) :

	avgcorrect = [0.0] * 3
	cases = [0.0] * 3

	for i, (inputs, labels) in enumerate(dataloader):
		# initialize
		torch.cuda.empty_cache()

		# forward
		inputs = inputs.half().cuda()
		labels = labels.cuda()
		overall_outputs = overall_model(inputs)
		single_outputs = single_model(inputs)
		del inputs

		
		# calculate accuracy
		for i in range(len(overall_outputs)) :
			if single_outputs[i] == 1 :
				overall_outputs[i] = 1
			if overall_outputs[i] == labels[i] :
				avgcorrect[labels[i]] += 1
			cases[labels[i]] += 1
		del labels, overall_outputs, single_outputs

	# print result
	print('\tOverall :', '%.2f%%'%(sum(avgcorrect) / sum(cases) * 100))
	for i in range(3) :
		print('\t\tclass', '%2d'%i, '%5d'%cases[i], '%5.2f%%'%(avgcorrect[i] / cases[i] * 100))

	# return accuracy
	return sum(avgcorrect) / sum(cases)


if __name__ == '__main__' :

	transform_test = transforms.Compose([
		transforms.Resize((256, 256)),
	    transforms.ToTensor(),
	    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	validationset = torchvision.datasets.ImageFolder(root = 'C:\\Users\\Frank\\Machine Learning\\mango\\dataset\\validation\\', transform = transform_test)
	validationloader = torch.utils.data.DataLoader(validationset, batch_size = 128, shuffle = False, num_workers = 10, pin_memory = True)

	with torch.no_grad() :
		forward(validationloader)