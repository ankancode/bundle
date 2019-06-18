import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train(model, **kwargs):
	criterion = nn.NLLLoss()
	optimizer = kwargs['optimizer']
	model.to(device)
	epochs = kwargs['epochs']                                                                      
	steps = 0
	running_loss = 0
	print_every = 5

	for epoch in range(epochs):
		for images, labels in kwargs['trainloader']:
			steps += 1
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()

			logps = model(images)
			loss = criterion(logps, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			print("Epoch: {}/{}.. ".format(epoch+1, epochs),
					"Training Loss: {:.3f}".format(running_loss/len(trainloader)))

			if kwargs['validation'] == True :
				if (steps % print_every) == 0:
					model.eval()
					validation_loss = 0
					accuracy = 0

					for images, labels in kwargs['validationloader']:
						images, labels = images.to(device), labels.to(device)
						logps = model(images)
						loss = criterion(logps, labels)
						validation_loss += loss.item()
						ps = torch.exp(logps)
						top_ps, top_class = ps.topk(1, dim=1)
						equality = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

					print("Validation Loss: {:.3f}".format(validation_loss/len(validationloader)),
							"Accuracy: {:.3f}".format(accuracy/len(validationloader)))

def model_test(model, **kwargs):