import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
	def __init__(self, resnet):
		super(Decoder, self).__init__()
		low_level_inplanes = 256
		self.resnet = resnet

		self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(48)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
										nn.BatchNorm2d(256),
										nn.ReLU(),
										nn.Dropout(0.5),
										nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
										nn.BatchNorm2d(256),
										nn.ReLU(),
										nn.Dropout(0.1),
										nn.Conv2d(256, 15, kernel_size=1, stride=1),
										nn.Sigmoid(),
										)
		self.last_deconv = nn.Sequential(nn.ConvTranspose2d(304, 128, kernel_size=4, stride=2, padding=1),
									nn.BatchNorm2d(128),
									nn.ReLU(),
									nn.Dropout(0.5),
									nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
									nn.BatchNorm2d(32),
									nn.ReLU(),
									nn.Dropout(0.1),
									)


		self._init_weight()


	def forward(self, x, low_level_feat): # 512, 64
		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat) 

		low_level_feat = self.maxpool(low_level_feat) 

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		# TODO : check ouyput size
		x = torch.cat((x, low_level_feat), dim=1) 
		if self.resnet == 'depth':
			x = self.last_deconv(x)
		elif self.resnet == 'heat':
			x = self.last_conv(x)
		else:
			raise NotImplementedError


		return x

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()