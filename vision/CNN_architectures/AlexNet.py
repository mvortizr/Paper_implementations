import torch
import torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self,in_channels=3,num_classes=1000):
    super().__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels= 96 , kernel_size=11, padding=0,stride=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(num_features=96),
      nn.Conv2d(in_channels=96, out_channels= 256 , kernel_size=5, padding=2,stride=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.BatchNorm2d(num_features=256),
      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1,stride=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1,stride=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1,stride=1),
      nn.ReLU(inplace=True), 
      nn.MaxPool2d(kernel_size=3, stride=2)    
		)
    self.linear_layers = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes)
    )

  def forward(self,x):
    out = self.conv_layers(x)
    out = out.reshape(out.shape[0],-1) # Flatten the conv output
    out = self.linear_layers(out)
    return out 



