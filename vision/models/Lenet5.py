"""
Architecture : LeNet
Paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

"""
import torch.nn as nn
from bundle.vision.utility import operate

__all__ = ['model', 'train', 'pred']

class Lenet5(nn.module):

	def __init__(self, features, num_classes = 10, init_weights = True):
		super(Lenet5, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(5 * 5 * 16, 120),
			nn.Tanh(True),
			nn.Linear(120, 84),
			nn.Tanh(True),
			nn.Linear(84, num_classes)
			nn.LogSoftmax(dim=1)
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules:
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				m.init.data.fill_(0.01)

	def make_layers(cfg, in_channels = 3):
		layers = []
		for v in cfg:
			if v == 'A':
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=5*5)
				layers += [conv2d, nn.Tanh(True)]
				in_channels = v
		return nn.Sequential(*layers)


def model(self, pretrained=False, init_weights=True):
	cfg = [6, 'A', 16, 'A']
	self.pretrain_flag = pretrained
	if pretrained:
		init_weights = False
	self.net = Lenet5(self.make_layers(cfg), init_weights)
	if pretrained:
		state_dict = load_state_dict_from_url(model_urls[lenet5], progress=True)
		self.net.load_state_dict(state_dict)
	return self.net

def train(self, **kwargs):
	if self.pretrain_flag == False:
		return operate.model_train(self.net, **kwargs)
	print("Model is already pretrained !!!")
	return self.net

def pred(self, image):
	pass
	return