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
from torchvision.transforms import InterpolationMode
from model import get_pose_net

eps=1e-10
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


"""
15 res block
32x32 heatmap
256x256 input
l2 loss final output
11, 14 layer supervision

distance module 2 res block + 1 fc layer
distance module 13, 15
"""
BN_MOMENTUM = 0.1
beta = torch.tensor([-0.05631891929412012, -0.0038333424842925286,
                -0.00024681888617308917, -0.00012153386798050158]) 

class mo2cap_ours(nn.Module):
	def __init__(self,pretrained_path=None) -> None:
		super().__init__()
		self.depthmap_module = get_pose_net('depth')
		self.mo2cap2_modele = mo2capmodel()
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			self.load_state_dict(tempmodel_ckpt)

	def forward(self,x):
		image_ = x['image']
		depth_feat=self.depthmap_module(image_)

		res_dict = self.mo2cap2_modele(x,depth = depth_feat)
		res_dict['depthmap'] = depth_feat['depthmap']
		res_dict['silhouette'] = depth_feat['silhouette']
		return res_dict

class mo2capmodel(nn.Module):
	def __init__(self,pretrained_path=None) -> None:
		super().__init__()
		self.inplanes = 64
		block = BasicBlock
		BatchNorm = nn.BatchNorm2d

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1_block_1_3 = self._make_layer(block,64,3,stride=1,BatchNorm=BatchNorm)
		self.layer2_block_4_7 = self._make_layer(block,128,4,stride=2,BatchNorm=BatchNorm)
		self.layer3_block_8_11 = self._make_layer(block,256,4,stride=1,BatchNorm=BatchNorm) # block 11 output supervision
		self.layer3_block_12_13 = self._make_layer(block,256,2,stride=2,BatchNorm=BatchNorm) # block 13 output dist 
		self.layer4_block_14 = self._make_layer(block,512,1,stride=1,BatchNorm=BatchNorm) # block 14 output supervision
		self.layer4_block_15 = self._make_layer(block,512,1,stride=2,BatchNorm=BatchNorm) # block 15 output dist 
		
		self.conv1_zoom = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1_zoom = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu_zoom = nn.ReLU(inplace=True)
		self.maxpool_zoom = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.inplanes = 64
		self.layer1_block_1_3_zoom = self._make_layer(block,64,3,stride=1,BatchNorm=BatchNorm)
		self.layer2_block_4_7_zoom = self._make_layer(block,128,4,stride=2,BatchNorm=BatchNorm)
		self.layer3_block_8_11_zoom = self._make_layer(block,256,4,stride=2,BatchNorm=BatchNorm) # block 11 output supervision
		self.layer3_block_12_13_zoom = self._make_layer(block,256,2,stride=1,BatchNorm=BatchNorm) # block 13 output dist 
		self.layer4_block_14_zoom = self._make_layer(block,512,1,stride=2,BatchNorm=BatchNorm) # block 14 output supervision
		self.layer4_block_15_zoom = self._make_layer(block,512,1,stride=1,BatchNorm=BatchNorm) # block 15 output dist 

		self.distance_module = self._make_dist_module(block,1536,2)

		self.supervision_1_conv = nn.Sequential(
			nn.Conv2d(256,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)
		self.supervision_2_conv = nn.Sequential(
			nn.Conv2d(512,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)
		self.supervision_1_zoom_conv = nn.Sequential(
			nn.Conv2d(256,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)
		self.supervision_2_zoom_conv = nn.Sequential(
			nn.Conv2d(512,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)

		self.last_conv = nn.Sequential(
			nn.Conv2d(512,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)

		self.last_conv_zoom = nn.Sequential(
			nn.Conv2d(512,15,kernel_size=1,stride=1),
			nn.BatchNorm2d(15),
			nn.Sigmoid(),
		)

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		
		self.get_d = nn.Linear(1536,16)
		# self.get_head = nn.Conv1d(15,1,kernel_size=1)
		self.relu = nn.ReLU()
	
		self.coord_linear = nn.Sequential(
			nn.Flatten(),
			nn.Linear(30,512),
			nn.Linear(512,512),
		)

		self.get_kd = nn.Sequential(
			nn.Linear(1536,4),
			nn.Tanh()
		)

		self.get_3d_coord = nn.Linear(784,48)
		# self.get_3d_coord = nn.Linear(528,48)


		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			self.load_state_dict(tempmodel_ckpt)
	def forward(self,x,depth=None):
		
		image = x['image']
		batch_size = len(image)

		image = self.conv1(image)
		image = self.bn1(image)
		image = self.relu(image)
		image = self.maxpool(image)
		
		image = self.layer1_block_1_3(image) # ->10,64,64,64
		image = self.layer2_block_4_7(image) # 
		image = self.layer3_block_8_11(image) # 256
		supervision_1 = image
		image = self.layer3_block_12_13(image) # 256
		dist_input_1 = image
		image = self.layer4_block_14(image) # 512
		supervision_2 = image
		image = self.layer4_block_15(image) # 512
		dist_input_2 = image

		image_zoom = x['image_zoom']

		image_zoom = self.conv1(image_zoom)
		image_zoom = self.bn1(image_zoom)
		image_zoom = self.relu(image_zoom)
		image_zoom = self.maxpool(image_zoom)
		
		image_zoom = self.layer1_block_1_3(image_zoom)
		image_zoom = self.layer2_block_4_7(image_zoom)
		image_zoom = self.layer3_block_8_11(image_zoom)
		supervision_1_zoom = image_zoom
		image_zoom = self.layer3_block_12_13(image_zoom)
		dist_input_1_zoom = image_zoom
		image_zoom = self.layer4_block_14(image_zoom)
		supervision_2_zoom = image_zoom
		image_zoom = self.layer4_block_15(image_zoom)
		dist_input_2_zoom = image_zoom
		
		dist_input_1 = F.interpolate(dist_input_1,size=(8,8),  mode='bilinear', align_corners=True) # 256
		# dist_input_2 = self.avgpool(dist_input_2) # 512
		dist_input_1_zoom = F.interpolate(dist_input_1,size=(8,8),  mode='bilinear', align_corners=True) # 256
		# dist_input_2_zoom = self.avgpool(dist_input_2_zoom)	# 512

		dist_feat = torch.cat([dist_input_1,dist_input_2,dist_input_1_zoom,dist_input_2_zoom],1)

		supervision_1 = self.supervision_1_conv(supervision_1)
		supervision_1 = F.interpolate(supervision_1,size=(32,32),  mode='bilinear', align_corners=True)
		
		supervision_2 = self.supervision_2_conv(supervision_2)
		supervision_2 = F.interpolate(supervision_2,size=(32,32),  mode='bilinear', align_corners=True)
		
		supervision_1_zoom = self.supervision_1_zoom_conv(supervision_1_zoom)
		supervision_1_zoom = F.interpolate(supervision_1_zoom,size=(32,32),  mode='bilinear', align_corners=True)
		
		supervision_2_zoom = self.supervision_2_zoom_conv(supervision_2_zoom)
		supervision_2_zoom = F.interpolate(supervision_2_zoom,size=(32,32),  mode='bilinear', align_corners=True)

		image = self.last_conv(image)
		heatmap = F.interpolate(image,size=(32,32),  mode='bilinear', align_corners=True)
		
		image_zoom = self.last_conv_zoom(image_zoom)
		heatmap_zoom = F.interpolate(image_zoom,size=(32,32),  mode='bilinear', align_corners=True)

		average_heatmap = self.average_heatmaps(heatmap,heatmap_zoom)

		# max_values, max_indices = torch.max(average_heatmap.view(average_heatmap.size(0), 15, -1), dim=2)
		max_values, max_indices = torch.max(average_heatmap.view(average_heatmap.size(0), 15, -1), dim=2)
		# 1D 인덱스를 2D (x, y) 좌표로 변환합니다.
		y_coords = (max_indices // 32)/10
		x_coords = (max_indices % 32)/10

		# 결과를 (배치 크기, 15, 2) 크기의 텐서로 반환합니다.
		joint_coords_2d = torch.stack((x_coords, y_coords), dim=2)

		d = self.distance_module(dist_feat)
		d = self.avgpool(d)
		d = d.view(batch_size,-1)
		d = self.get_d(d)
		d = self.relu(d)
		joint_ = self.coord_linear(joint_coords_2d)

		if depth is not None:
			depth_embed = depth['embed_feature'].detach()
			depth_embed = self.avgpool(depth_embed)
			depth_embed = depth_embed.view(batch_size,-1) # 256

			total_feat = torch.cat([joint_,d,depth_embed],1).view(batch_size,-1)
		else:
			total_feat = torch.cat([joint_,d],1).view(batch_size,-1)
		
		joint_coord_3d = self.get_3d_coord(total_feat).view(batch_size,16,3)
		
		# magnitude = torch.sqrt((joint_coords_3d_vec ** 2).sum(dim=2))

   		#  # Compute the scaling factor
		# scale_factor = (d / magnitude)+eps

		# # Apply the scaling factor to each 3D vector
		# joint_coord_3d = joint_coords_3d_vec * scale_factor.unsqueeze(2)

		# joint_coord_3d = self.compute_3d_joint_position(joint_coords_2d,d,kd)
		
		# head_coord = self.get_head(joint_coord_3d)
		# joint_coord_3d = torch.cat((joint_coord_3d[:,:1,:],head_coord,joint_coord_3d[:,1:,:]),dim=1)


		res ={
			'fully_heatmap_1':supervision_1,
			'fully_heatmap_2':supervision_2,
			'zoom_heatmap_1':supervision_1_zoom,
			'zoom_heatmap_2':supervision_2_zoom,
			'pred_pose':joint_coord_3d,
			'heatmap':heatmap,
			'heatmap_zoom':heatmap_zoom,
			'average_heatmap':average_heatmap,
		}
		return res


	
	def compute_rho(self, u, v):
		return torch.sqrt(u**2 + v**2)

	def f(self, rho, beta):
		return beta[0] + beta[1]*rho + beta[2]*rho**2 + beta[3]*rho**3  # Extend as needed

	def map_uv_to_xyz(self, u, v, beta):
		rho = self.compute_rho(u, v)
		z = self.f(rho, beta)
		return u, v, z

	def compute_3d_joint_position(self, uv, d, beta):
		x, y, z = self.map_uv_to_xyz(uv[..., 0], uv[..., 1], beta)
		norm = torch.sqrt(x**2 + y**2 + z**2)
		scale_factor = ((d / norm)+eps).view(d.shape[0], d.shape[1], 1)
		return torch.stack([x, y, z], dim=-1) * scale_factor
	
	def average_heatmaps(self,heatmap_A, heatmap_B):
		"""
		Average two heatmaps: one from the original image and one from the zoomed and center-cropped image.
		heatmap_A: Heatmap from the original image.
		heatmap_B: Heatmap from the zoomed and center-cropped image.
		"""
		
		# Heatmap B'의 축소
		h, w = heatmap_A.shape[-2], heatmap_A.shape[-1]
		resized_heatmap_B = F.interpolate(heatmap_B, size=(h // 2, w // 2),  mode='bilinear', align_corners=True)

		# Padding 추가
		padding = (h // 4, w // 4, h // 4, w // 4)  # left, top, right, bottom
		padded_heatmap_B = F.pad(resized_heatmap_B, padding, "constant", 0)

		# Create a mask for the padded area of heatmap_B
		mask = torch.ones_like(padded_heatmap_B)
		mask[..., :h//4, :] = 0
		mask[..., -h//4:, :] = 0
		mask[..., :, :w//4] = 0
		mask[..., :, -w//4:] = 0

		# Compute the averaged heatmap
		averaged_heatmap = (heatmap_A + padded_heatmap_B ) / (1 + mask)
			
		return averaged_heatmap

	def _make_layer(self, block, planes, blocks, stride=1, BatchNorm=None):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
							kernel_size=1, stride=stride, bias=False),
				BatchNorm(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)
	
	def _make_dist_module(self,block, planes, blocks, stride=1):
		downsample = None
		layers = []
		layers.append(block(planes, planes, stride))
		layers.append(block(planes, planes, stride))
		# layers.append(nn.AdaptiveAvgPool2d((1,1)))
		# layers.append(nn.Linear(planes,1))
		# layers.append(nn.ReLU())
		return nn.Sequential(*layers)



	

