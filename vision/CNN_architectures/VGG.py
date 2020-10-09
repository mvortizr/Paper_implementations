import torch
import torch.nn as nn

#References 
#Using this hack https://www.youtube.com/watch?v=ACmuBbuXn20  because I am lazy and don't want to do each layer by itself
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

VGG_arch = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
	
	def __init__(self, in_channels=3,num_classes=1000,vgg_type='D'):
		
		super(VGG_net,self).__init__()
		
		self.in_channels=in_channels
		self.num_classes=num_classes
		self.conv_layers = self.create_conv_layers(VGG_arch[vgg_type])
		self.fc_layers = self.create_fc_layers()

	def forward(self, x):
		out = self.conv_layers(x)
		out = out.reshape(out.shape[0],-1)# Flatten the conv output
		out = self.fc_layers(out)
		return out

	def create_fc_layers(self):
		#output from the convolutional has shape 512*7*7 after all the maxpools
		return nn.Sequential(
			nn.Linear(512 * 7 * 7,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,self.num_classes)
		)


	def create_conv_layers(self,architecture):
		layers =[]
		in_channels = self.in_channels

		for layer in architecture:

			if layer == 'M': # It's a MaxPool layer
				layers +=  [nn.MaxPool2d(kernel_size=2, stride=2)]
			else: # It's a Conv + ReLU
				conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
				layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = layer

		return nn.Sequential(*layers)
            




