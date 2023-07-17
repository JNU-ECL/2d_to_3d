import math
import logging
from collections import OrderedDict
from yacs.config import CfgNode as CN
from easydict import EasyDict
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from smplx.body_models import SMPL
import numpy as np
from collections import OrderedDict
from geometry import rot6d_to_rotmat, projection_temp, rotation_matrix_to_angle_axis, create_euler
import torch.utils.model_zoo as model_zoo
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from model import get_pose_net

BN_MOMENTUM = 0.1

class xR_egoposemodel(nn.Module):
	def __init__(self,pretrained_path = None):
		super().__init__()
		BatchNorm = nn.BatchNorm2d
		self.heatmap_module = get_pose_net('heatmap')
		self.depthmap_module = get_pose_net('depth')
		self.dual_branch_module = Dual_Branch()
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			self.load_state_dict(tempmodel_ckpt)

	def forward(self,x):
		image_ = x['image']
		depth_feat=self.depthmap_module(image_)
		heatmap_feat=self.heatmap_module(image_)

		res_dict = self.dual_branch_module(heat = heatmap_feat,depth = depth_feat)
		res_dict['depthmap'] = depth_feat['depthmap']
		res_dict['silhouette'] = depth_feat['silhouette']
		return res_dict
		

	
class Dual_Branch(nn.Module):
	def __init__(self) -> None:
		super().__init__()	
		self.deconv_block_0 = nn.Sequential(
			nn.ConvTranspose2d(in_channels=2048,
					  out_channels=15,
					  kernel_size=4,
					  stride=2,
					  padding=1),
			nn.BatchNorm2d(15, momentum=BN_MOMENTUM),
			nn.Sigmoid(),# .2 leakiness
		)
		self.conv_block_0 = self._make_conv_block(15,64)
		self.conv_block_1 = self._make_conv_block(64,128)
		self.conv_block_2 = self._make_conv_block(128,256)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.forward_linear = self._make_linear(256)
		self.pose_linear = self._make_dual_linear_pose()
		self.heat_linear = self._make_dual_linear_heat()
		self.deconv_block = self._make_deconv_block(18432)
		
		self.depth_forward = nn.Sequential(
			nn.Linear(256, 512),
			nn.BatchNorm1d(512,momentum=BN_MOMENTUM),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512,momentum=BN_MOMENTUM),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 20),
		)

		self.conv_block_0.apply(self._weights_init)
		self.conv_block_1.apply(self._weights_init)
		self.conv_block_2.apply(self._weights_init)
		self.forward_linear.apply(self._weights_init)
		self.pose_linear.apply(self._weights_init)
		self.heat_linear.apply(self._weights_init)
		self.deconv_block.apply(self._weights_init)

		
	def forward(self,heat,depth):
		h_raw = heat['heatmap']
		h = heat['embed_feature'].detach()
		d = depth['embed_feature'].detach()

		batch_size = len(h)
		
		conv_feature = self.conv_block_0(h_raw) # -> torch.Size([10, 64, 23, 23])
		conv_feature = self.conv_block_1(conv_feature) # -> torch.Size([10, 128, 11, 11])
		conv_feature = self.conv_block_2(conv_feature) 
		conv_feature = self.avgpool(conv_feature) # -> torch.Size([10, 256, 1, 1])
		conv_feature = conv_feature.view(batch_size,-1)

		h = self.avgpool(h)
		h = h.view(batch_size,-1)
		d = self.avgpool(d)
		d = d.view(batch_size,-1)

		# total_feat = torch.cat([h], 1)

		linear_feature = self.forward_linear(conv_feature)
		depth_feature = self.depth_forward(d)
		total_feat = torch.cat([linear_feature,depth_feature],1)
		pose = self.pose_linear(total_feat).view(-1,16,3) # TODO : concat 
		
		heat_feature = self.heat_linear(linear_feature)

		heat_feature = heat_feature.view(batch_size,-1,1,1)
		
		pred_2nd_heatmap = self.deconv_block(heat_feature)
		pred_2nd_heatmap = F.interpolate(pred_2nd_heatmap,size=(47,47),mode='bilinear',align_corners=True)
		
		pred_normal = self._get_normal(pose)

		return {
			'pred_pose':pose,
			'pred_1st_heatmap':h_raw,
			'pred_2nd_heatmap':pred_2nd_heatmap,
			'pred_normal':pred_normal,
		}

	def _weights_init(self, m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight, gain=0.01)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

	def _get_normal(self,pose):
		p_idx = torch.tensor([0,1,1,2,3,1,5,6,1,8,9,10,1,12,13,14]).to(pose.device)
		pose_p = torch.index_select(pose,1,p_idx)
		normal = pose - pose_p
		return normal

	def _make_linear(self,in_dim):
		return nn.Sequential(
			nn.Linear(in_dim,18432),
			nn.BatchNorm1d(18432,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(18432,2048),
			nn.BatchNorm1d(2048,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(2048,512),
			nn.BatchNorm1d(512,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(512,20),
			nn.BatchNorm1d(20,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
		)
	
	def _make_dual_linear_pose(self):
		return nn.Sequential(
			nn.Linear(40,32),
			nn.BatchNorm1d(32,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(32,32),
			nn.BatchNorm1d(32,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(32,48),
		)

	def _make_dual_linear_heat(self):
		return nn.Sequential(
			nn.Linear(20,512),
			nn.BatchNorm1d(512,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(512,2048),
			nn.BatchNorm1d(2048,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
			nn.Linear(2048,18432),
			nn.BatchNorm1d(18432,momentum=BN_MOMENTUM),
			nn.LeakyReLU(),
		)


	def _make_conv_block(self,in_channels, out_channels, kernel_size=4, stride=2, padding=1):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels,
					  out_channels=out_channels,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
			nn.LeakyReLU(),# .2 leakiness
			# nn.Dropout(p=0.2),
		)

	
	def _make_deconv_block(self,in_channels, kernel_size=4, stride=2, padding=1):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels=in_channels,
					  out_channels=512,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
			nn.LeakyReLU(),# .2 leakiness
			nn.ConvTranspose2d(in_channels=512,
					  out_channels=128,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
			nn.LeakyReLU(),# .2 leakiness
			nn.ConvTranspose2d(in_channels=128,
					  out_channels=64,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
			nn.LeakyReLU(),# .2 leakiness
			nn.ConvTranspose2d(in_channels=64,
					  out_channels=15,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(15, momentum=BN_MOMENTUM),
			nn.Sigmoid(),# .2 leakiness
		)





class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = BatchNorm(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   dilation=dilation, padding=dilation, bias=False)
		self.bn2 = BatchNorm(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = BatchNorm(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.dilation = dilation

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
		self.inplanes = 64
		super(ResNet, self).__init__()
		blocks = [1, 2, 4]
		if output_stride == 16:
			#strides = [1, 1, 1, 1]
			strides = [1, 2, 2, 1]
			dilations = [1, 1, 1, 2]
		elif output_stride == 8:
			strides = [1, 2, 1, 1]
			dilations = [1, 1, 2, 4]
		else:
			raise NotImplementedError

		# Modules
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
								bias=False)
		self.bn1 = BatchNorm(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
		self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
		# self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
		self._init_weight()

		if pretrained:
			self._load_pretrained_model()

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				BatchNorm(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

		return nn.Sequential(*layers)

	def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				BatchNorm(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
							downsample=downsample, BatchNorm=BatchNorm))
		self.inplanes = planes * block.expansion
		for i in range(1, len(blocks)):
			layers.append(block(self.inplanes, planes, stride=1,
								dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

		return nn.Sequential(*layers)

	def forward(self, input):
		x = self.conv1(input)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		low_level_feat = x
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x, low_level_feat

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _load_pretrained_model(self):
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
		pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
		
		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

def ResNet101(output_stride, BatchNorm, pretrained=True):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	#model = ResNet(Bottleneck, [2, 2, 2, 2], output_stride, BatchNorm, pretrained=pretrained)
	#model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
	model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
	return model