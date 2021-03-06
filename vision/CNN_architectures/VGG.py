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
	
	def __init__(self, in_channels=3,num_classes=1000,vgg_type='D',output_conv_layers=512 * 7 * 7,init_weights=True):
		#Output from the convolutional has shape 512*7*7 after all the maxpools for ImageNet (3x224x224)
		super(VGG_net,self).__init__()
		
		self.in_channels=in_channels
		self.num_classes=num_classes
		self.output_conv_layers=output_conv_layers
		self.conv_layers = self.create_conv_layers(VGG_arch[vgg_type])
		self.fc_layers = self.create_fc_layers()
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		out = self.conv_layers(x)
		out = out.reshape(out.shape[0],-1)# Flatten the conv output
		out = self.fc_layers(out)
		return out

	def create_fc_layers(self):
		
		return nn.Sequential(
			nn.Linear(self.output_conv_layers,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,self.num_classes)
		)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


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
            




