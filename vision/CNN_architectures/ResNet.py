import torch
import torch.nn as nn



class block(nn.Module):
  
  # Reusable block of ResNet, valid for all the ResNet arch with 50+ layers

  def __init__(self,in_channels,out_channels,downsample=None,conv2_stride=1,expansion):
    
    super(block,self).__init__()
    
    self.expansion = expansion # expansion x outchannels = filters of the last layer of the block

    #ReLU layer
    self.relu = nn.ReLU(inplace=True)
    
    #conv1 1x1, outchannels
    self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    
    #conv2 3x3, outchannels
    self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=conv2_stride,padding=1)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    #conv3 1x1, outchannels x self.expansion
    self.conv3 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
    self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)

    # Identity downsample
    self.downsample = downsample

  def forward(self,x): 
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    if downsample is not None:
      identity = self.downsample(identity)

    x+=identity
    x = self.relu(x)
    
    return x

class ResNet(nn.Module):
  def __init__(self,block,layers,image_channels,num_classes, expansion):
    # layers = A list with the number of times to repeat each block. E.g ResNet50 is [3,4,6,3]
    super(ResNet,self).__init__()
    
    self.in_channels = 64
    self.expansion = expansion

    # Initial layers
    self.conv1 = nn.Conv2d(in_channels=image_channels,out_channels=self.in_channels,kernel_size=7,stride=2,padding=3)
    self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

    # ResNet layers, remember output = out_channel * expansion
    self.layer1 = self._make_layer(block,layer[0],out_channels=64,stride=1)
    self.layer2 = self._make_layer(block,layer[1],out_channels=128,stride=2)
    self.layer3 = self._make_layer(block,layer[2],out_channels=256,stride=2)
    self.layer4 = self._make_layer(block,layer[3],out_channels=512,stride=2)

    # Final layers
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(in_features=512*4,num_classes=num_classes)

  def forward(x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc(x)
    return x


  def generate_downsample(self,out_channels,stride):
    return nn.Sequential(
      nn.Conv2d(in_channels=self.in_channels,out_channels=out_channels*self.expansion,kernel_size=1,stride=stride),
      nn.BatchNorm2d(num_features=out_channels * self.expansion)
    )

  def _make_layer(self,block,num_residual_blocks,out_channels,stride):
    downsample = None
    layers = []

    if stride != 1 or self.in_channels != out_channels * self.expansion:
      downsample = self.generate_downsample(out_channels,stride)

    layers.append(
      block(self.in_channels,out_channels,downsample,stride,self.expansion)
    )

    self.in_channels = out_channels * self.expansion

    for i in range(num_residual_blocks -1):
      layers.append(
        block(self.in_channels,out_channels,expansion=self.expansion)
      )
    
    return nn.Sequential(*layers)

class ResNet50(img_channels=3,num_classes=1000):
  return ResNet(block,[3,4,6,3],img_channels,num_classes)

