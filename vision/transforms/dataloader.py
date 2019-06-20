import matplotlib.pyplot as plt 
import torch
from torchvision import datasets, transforms
import os
from PIL import Image

class Load_Dataset:

	def __init__(self, path):

		self.train_transforms = transforms.Compose([transforms.Resize(32),
			transforms.RandomRotation(30),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
			])


		self.val_test_transforms = transforms.Compose([transforms.Resize(32),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
			])	

	def find_classes(self, path):
	    classes = os.listdir(path)
	    classes.sort()
	    class_to_idx = {i:classes[i] for i in range(len(classes))}
	    return class_to_idx

	def getLoader(self, path, step):

		if step == 'validation' or step =='test':
			transform = self.val_test_transforms
		elif step == 'train':

			transform = self.train_transforms

		data_load_path = os.path.join(path,step)
		
		data = datasets.ImageFolder(data_load_path, transform = transform)
		dataloader = torch.utils.data.DataLoader(train_data, batch_size=32)

		return dataloader

	def image_loader(image_path):
		image = Image.open(image_path)
		image = self.val_test_transforms(image).float()
		image = torch.tensor(image, requires_grad=True)
		image = image.unsqueeze(0)
		return image
