
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

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



# SMPL_MEAN_PARAMS = '/workspace/2d_to_3d/model/smpl_mean_params.npz'
# SMPL_MODEL_DIR = '/workspace/2d_to_3d/model/smpl'
# use_cuda = torch.cuda.is_available()


class TempModel(nn.Module):
	def __init__(self,pretrained_path:str =None, load_heatmap=True, load_depthmap=True, load_regressor=True):
		super().__init__()
		load_heatmap=load_heatmap
		load_depthmap=load_depthmap
		load_regressor=load_regressor
		self.heatmap_module = get_pose_net('heatmap')
		self.depthmap_module = get_pose_net('depth')
		self.regressor=Regressor(depthmapfeat_c=256,heatmapfeat_c=256)
		
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			if load_heatmap and load_depthmap and load_regressor:
				self.load_state_dict(tempmodel_ckpt)
				if not load_regressor:
					heatmap_state_dict = OrderedDict()
					depthmap_state_dict = OrderedDict()
					if load_heatmap and load_depthmap:
						for k,v in tempmodel_ckpt.items():
							if k.startswith('heatmap_module.'):
								new_key = k[len('heatmap_module.'):]
								heatmap_state_dict[new_key] = v
							if k.startswith('depthmap_module.'):
								new_key = k[len('depthmap_module.'):]
								depthmap_state_dict[new_key] = v
						self.heatmap_module.load_state_dict(heatmap_state_dict)
						self.depthmap_module.load_state_dict(depthmap_state_dict)
					elif load_heatmap:
						for k,v in tempmodel_ckpt.items():
							if k.startswith('heatmap_module.'):
								new_key = k[len('heatmap_module.'):]
								heatmap_state_dict[new_key] = v
						self.heatmap_module.load_state_dict(heatmap_state_dict)
					else:
						for k,v in tempmodel_ckpt.items():
							if k.startswith('depthmap_module.'):
								new_key = k[len('depthmap_module.'):]
								depthmap_state_dict[new_key] = v
						self.depthmap_module.load_state_dict(depthmap_state_dict)
				
	def forward(self, x):
		res = {}
		image_ = x['image']
	
		
		depth_feat=self.depthmap_module(image_)
		heatmap_feat=self.heatmap_module(image_)

		regressor_res_dict=self.regressor(
			heatmap_feat['embed_feature'],
			heatmap_feat['heatmap'],
			depth_feat['embed_feature'].detach(),
			depth_feat['depthmap'].detach(),
		)




		res.update({
   	  		'regressor_dict' : regressor_res_dict,
			'pred_pose' : regressor_res_dict['pred_joint'],
	 		'depthmap' : depth_feat['depthmap'],
			'silhouette' : depth_feat['silhouette'],
	 		'heatmap' : heatmap_feat['heatmap'],
		})    
		return res



args=EasyDict({
	"cfg_file":r'/workspace/2d_to_3d/model/config/384x288_d256x3_adam_lr1e-3.yaml',
	"misc":None
})



cfg = CN(new_allowed=True)

def get_cfg_defaults():
	"""Get a yacs CfgNode object with default values for my_project."""
	# Return a clone so that the defaults will not be altered
	# This is for the "local variable" use pattern
	# return cfg.clone()
	return cfg

def update_cfg(cfg_file):
	# cfg = get_cfg_defaults()
	cfg.merge_from_file(cfg_file)
	# return cfg.clone()
	return cfg

def parse_args(args):
	cfg_file = args.cfg_file
	if args.cfg_file is not None:
		cfg = update_cfg(args.cfg_file)
	else:
		cfg = get_cfg_defaults()

	if args.misc is not None:
		cfg.merge_from_list(args.misc)

parse_args(args)

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out



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


class Decoder(nn.Module):
	def __init__(self, outplanes, resnet):
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
										nn.Conv2d(256, outplanes, kernel_size=1, stride=1),
										nn.Sigmoid()
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
		low_level_feat = self.relu(low_level_feat) # 48

		low_level_feat = self.maxpool(low_level_feat) 

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		# TODO : check ouyput size
		x = torch.cat((x, low_level_feat), dim=1) 
		if self.resnet == 'depth':
			x = self.last_deconv(x)
		elif self.resnet == 'heatmap':
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
			

class _AtrousModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
		
class WASP(nn.Module):
	def __init__(self):
		super(WASP, self).__init__()
		self.output_stride = 16
		inplanes = 2048 #resnet34:512, resnet50:2048
		if self.output_stride == 16:
			dilations = [ 1,2,3,4]
			# dilations = [24, 18, 12,  6]
			#dilations = [6, 6, 6, 6]
		elif self.output_stride == 8:
			dilations = [48, 36, 24, 12]


		self.aspp1 = _AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
		self.aspp2 = _AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1])
		self.aspp3 = _AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2])
		self.aspp4 = _AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3])

		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
												nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
												nn.BatchNorm2d(256),
												nn.ReLU())

		self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
		self.conv2 = nn.Conv2d(256,256,1,bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)
		
		self._init_weight()

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# m.weight.data.normal_(0, math.sqrt(2. / n))
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, x):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x1)
		x3 = self.aspp3(x2)
		x4 = self.aspp4(x3)

		x1 = self.conv2(x1)
		x2 = self.conv2(x2)
		x3 = self.conv2(x3)
		x4 = self.conv2(x4)
		
		x1 = self.conv2(x1)
		x2 = self.conv2(x2)
		x3 = self.conv2(x3)
		x4 = self.conv2(x4)

		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1) # -> [10, 1280, 16, 16]

		x = self.conv1(x) # -> [10, 256, 16, 16]
		x = self.bn1(x)
		x = self.relu(x)

		return self.dropout(x) 

class PoseResNet_depth(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		super(PoseResNet_depth, self).__init__()
		self.inplanes = 64
		self.output_stride =16
		self.pretrained = True
		BatchNorm = nn.BatchNorm2d
		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		self.wasp = WASP()
		self.decoder = Decoder(1,'depth')
		self.decoder_ = Decoder(1,'depth') # if sep decoder state is on remove #

		blocks = [1, 2, 4]
		if self.output_stride == 16:
			#strides = [1, 1, 1, 1]
			strides = [1, 2, 2, 1]
			dilations = [1, 1, 1, 2]
		elif self.output_stride == 8:
			strides = [1, 2, 1, 1]
			dilations = [1, 1, 2, 4]
		else:
			raise NotImplementedError

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
		self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

		

		self.deconv_depth = nn.Sequential(
			nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid(),
		)

		self.deconv_sil = nn.Sequential(
			nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
			# nn.Sigmoid(),
		)

		self.with_out_wasp = nn.Conv2d(2048,256,3)

		self._init_weight()
		if self.pretrained:
			self._load_pretrained_model()

	# def _load_pretrained_model(self):
	# 	#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
	# 	pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
	# 	# pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
	# 	# pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
		
	# 	model_dict = {}
	# 	state_dict = self.state_dict()
	# 	for k, v in pretrain_dict.items():
	# 		if k in state_dict:
	# 			model_dict[k] = v
	# 	state_dict.update(model_dict)
	# 	self.load_state_dict(state_dict)

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

	def _make_deconv_layer(self,in_planes, out_planes, kernel=4, stride=2, padding=1):
		return nn.Sequential(
			nn.ConvTranspose2d(
				in_channels=in_planes,
				out_channels=out_planes,
				kernel_size=kernel,
				stride=stride,
				padding=padding,
				bias=self.deconv_with_bias
							   ),
			nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			)
	
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

	
	def _load_pretrained_model(self):
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
		pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
		# pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
		
		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

	def forward(self, x): # -> 3x256x256
		x = self.conv1(x) # -> 64x128x128
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x) # -> 64x64x64

		low_level_feat = self.layer1(x) # ->256x64x64
		x_1 = self.layer2(low_level_feat) # ->512x32x32
		x_2 = self.layer3(x_1) # ->1024x16x16
		x = self.layer4(x_2) # ->2048x16x16

		# temp_x = self.with_out_wasp(x)
		temp_x = self.wasp(x) # -> 256x16x16
		"""
		x_= self.deconv_layer1(x_) #->128x16x16
		x_= self.deconv_layer2(x_) #->64x32x32
		x_= self.deconv_layer3(x_) #->32x64x64
		x_= self.deconv_layer4(x_+low_level_feat) #->32x64x64
		x_= self.deconv_layer5(x_) # 32+64= 96x64x64->
		"""
	 	# TODO : change interpolate to deconv only dep,silhouette
		x_ = self.decoder(temp_x,low_level_feat)
		x_sil = self.decoder_(temp_x,low_level_feat)
		x_depthmap = self.deconv_depth(x_)
		x_silhouette  = self.deconv_sil(x_)
		# x_dep = self.decoder(temp_x,low_level_feat)

		# final bilinear interpolation to reach the original input size
		depthmap = F.interpolate(x_depthmap, size=(256,256), mode='bilinear', align_corners=True)
		silhouette = F.interpolate(x_silhouette, size=(256,256), mode='bilinear', align_corners=True)


		res = {
			'depthmap':depthmap, 
			'silhouette' :silhouette,
			'embed_feature':temp_x,
		}

		return res

class PoseResNet_heatmap(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		super(PoseResNet_heatmap, self).__init__()
		self.inplanes = 64

		self.output_stride =16
		self.pretrained = True
		BatchNorm = nn.BatchNorm2d
		blocks = [1, 2, 4]
		if self.output_stride == 16:
			#strides = [1, 1, 1, 1]
			strides = [1, 2, 2, 1]
			dilations = [1, 1, 1, 2]
		elif self.output_stride == 8:
			strides = [1, 2, 1, 1]
			dilations = [1, 1, 2, 4]
		else:
			raise NotImplementedError


		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		self.wasp = WASP()
		self.decoder = Decoder(15,'heatmap')
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


		self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
		self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)




		"""
		self.deconv_layer1 = self._make_deconv_layer(512,256)
		self.deconv_layer2 = self._make_deconv_layer(256,128)
		self.deconv_layer3 = self._make_deconv_layer(128,64)
		self.deconv_layer4 = self._make_deconv_layer(64,32)
		self.deconv_layer4_ = self._make_deconv_layer(64,32)
		self.deconv_layer5 = nn.Sequential(
			nn.ConvTranspose2d(
				in_channels=32,
				out_channels=1,
				kernel_size=4,
				stride=2,
				padding=1,
				bias=self.deconv_with_bias
							   ),
			nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
			nn.Sigmoid(),
			)
		self.deconv_layer5_ = nn.Sequential(
			nn.ConvTranspose2d(
				in_channels=32,
				out_channels=1,
				kernel_size=4,
				stride=2,
				padding=1,
				bias=self.deconv_with_bias
							   ),
			nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True),
			)
		"""
		self.with_out_wasp = nn.Conv2d(2048,256,3)
		self._init_weight()
		if self.pretrained:
			self._load_pretrained_model()



	def _load_pretrained_model(self):
		#pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
		# pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
		pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
		# pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
		
		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

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

	def _make_deconv_layer(self,in_planes, out_planes, kernel=4, stride=2, padding=1):
		return nn.Sequential(
			nn.ConvTranspose2d(
				in_channels=in_planes,
				out_channels=out_planes,
				kernel_size=kernel,
				stride=stride,
				padding=padding,
				bias=self.deconv_with_bias
							   ),
			nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True),
			)
	
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

	def forward(self, x): # -> 3x256x256
		x = self.conv1(x) # -> 64x128x128
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x) # -> 64x64x64

		low_level_feat = self.layer1(x) # -> 64x64x64
		x_1 = self.layer2(low_level_feat) # -> 128x32x32
		x_2 = self.layer3(x_1) # -> 256x16x16
		x = self.layer4(x_2) # -> 512x8x8

		temp_x = self.with_out_wasp(x)
		# temp_x = self.wasp(x)

		x_heatmap = self.decoder(temp_x,low_level_feat)
		heatmap = F.interpolate(x_heatmap, size=(64,64), mode='bilinear', align_corners=True)
		res = {
			'heatmap':heatmap,
			'embed_feature':temp_x,
		}
		return res


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
			   34: (BasicBlock, [3, 4, 6, 3]),
			   50: (Bottleneck, [3, 4, 6, 3]),
			   101: (Bottleneck, [3, 4, 23, 3]),
			   152: (Bottleneck, [3, 8, 36, 3])}

def get_pose_net(model_n=None, cfg=cfg,  **kwargs):
	num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
	# style = cfg.MODEL.STYLE

	# if style == 'caffe':
	#     block_class = Bottleneck_CAFFE
	if model_n == 'heatmap':
		block_class, layers = resnet_spec[50]
		model = PoseResNet_heatmap(block_class, layers, cfg, **kwargs)
	elif model_n == 'depth':
		block_class, layers = resnet_spec[50]
		model = PoseResNet_depth(block_class, layers, cfg, **kwargs)
	# if is_train and cfg.MODEL.INIT_WEIGHTS:
	# 	model.init_weights(cfg.MODEL.PRETRAINED)

	return model

class Regressor(nn.Module):
	def __init__(self, heatmapfeat_c, depthmapfeat_c ):
		super().__init__()
		njoint = 16 * 3 
		self.flatten = nn.Flatten()
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.bilinear_layer_pose = nn.ModuleList()


		for _ in range(2):
			block = self._make_bilinear(1024)
			self.bilinear_layer_pose.append(block)


		self.fc1 = nn.Linear(heatmapfeat_c + depthmapfeat_c + 128, 1024)
		self.bn1 = nn.BatchNorm1d(1024,momentum=BN_MOMENTUM)
		self.relu1 = nn.ReLU()
		self.drop1 = nn.Dropout()

		self.conv_block0 = self._make_block(1,32)
		self.conv_block1 = self._make_block(32,64) 
		self.conv_block2 = self._make_block(64,128) 

		self.conv_block1_ = self._make_block(15,64) 
		self.conv_block2_ = self._make_block(64,128) 

		self.decjoint = nn.Linear(1024, njoint)
		self.deccam_trans = nn.Linear(1024, 3)
		self.deccam_rot = nn.Linear(1024, 3)

		nn.init.xavier_uniform_(self.decjoint.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_trans.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_rot.weight, gain=0.01)

	def _make_block(self,in_channels, out_channels, kernel_size=4, stride=2, padding=1):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels,
					  out_channels=out_channels,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
		)
	
	def _make_bilinear(self,in_dim):
		return nn.Sequential(
			nn.Linear(in_dim, 1024),
			nn.BatchNorm1d(1024,momentum=BN_MOMENTUM),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024, 1024),
			nn.BatchNorm1d(1024,momentum=BN_MOMENTUM),
			nn.ReLU(),
			nn.Dropout(),
		)

		
	def forward(self, heatmap_embed=None, heatmap=None, depthmap_embed=None, depthmap=None):
		x=heatmap_embed # 256x8x8
		x2=depthmap_embed # 256x8x8
		x3=depthmap # 1x256x256
		x4=heatmap # 15x64x64
		# TODO : heatmap 입력 argmax로 차원당 1 하나씩
		batch_size = x.shape[0]

		x = self.avgpool(x) #256

		x2 = self.avgpool(x2) # 256

		x3 = self.conv_block0(x3) #
		x3 = self.conv_block1(x3) #
		x3 = self.conv_block2(x3) # -> 128
		x3 = self.avgpool(x3) 

		# x4 = self.conv_block1_(x4)
		# x4 = self.conv_block2_(x4)
		# x4 = self.avgpool(x4) 

		heatmap_e_feat = x.view(batch_size,-1)
		depth_e_feat = x2.view(batch_size,-1)
		depthmap_feat = x3.view(batch_size,-1) 
		# heatmap_feat = x4.view(batch_size,-1)

		total_feat = torch.cat([heatmap_e_feat, depth_e_feat, depthmap_feat], 1)

		pred_joint = self.fc1(total_feat)
		pred_joint = self.bn1(pred_joint)
		pred_joint = self.relu1(pred_joint)
		pred_joint = self.drop1(pred_joint)
		pose_residual = pred_joint

		for layer in self.bilinear_layer_pose:
			pred_joint = layer(pred_joint)
			pred_joint += pose_residual
			# pose_residual = pred_joint


		pred_joint = self.decjoint(pred_joint).view(-1,16,3)
	
		res = {
			# 'pred_trans' : pred_trans,
			# 'pred_rot' : pred_rot,
			'pred_joint': pred_joint,
		}
		return res
	
	def _make_block(self,in_channels, out_channels, kernel_size=4, stride=2, padding=1):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels,
					  out_channels=out_channels,
					  kernel_size=kernel_size,
					  stride=stride,
					  padding=padding),
			nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.2),
		)


	
class CustomActivation(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0, 1)