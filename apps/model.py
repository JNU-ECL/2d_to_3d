
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


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
# TODO : kaolin based renderer 구현할것 
# import kaolin as kal

SMPL_MEAN_PARAMS = '/workspace/2d_to_3d/model/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/workspace/2d_to_3d/model/smpl'
use_cuda = torch.cuda.is_available()


class TempModel(nn.Module):
	def __init__(self,pretrained_path:str = None):
		super().__init__()
		self.feature_model1 = get_pose_net(True)
		self.regressor2=Regressor2(1072,smpl_mean_params=SMPL_MEAN_PARAMS)
		
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			self.load_state_dict(tempmodel_ckpt)

	def forward(self, x, is_train,epoch):
		res = {}
		image_ = x['image']
		depth_ = x['depth']
		heatmap_ = x['heatmap']
		# trans_, rot_ = x['camera_info']
		
		feature_dict=self.feature_model1(image_)
	
		
		if is_train:
			if epoch<2:
				regressor2_res_dict=self.regressor2(
					heatmap_,
					depth_,
					)
			elif epoch<4:
				regressor2_res_dict=self.regressor2(
					feature_dict['heatmap'].detach(),
					feature_dict['depthmap'].detach(),
				)
			else:
				regressor2_res_dict=self.regressor2(
					feature_dict['heatmap'],
					feature_dict['depthmap'],
				)
		else:
			regressor2_res_dict=self.regressor2(
				feature_dict['heatmap'],
				feature_dict['depthmap'],
				)

		res.update({
   	  		'regressor2_dict' : regressor2_res_dict,
	 		'depth_feature' : feature_dict['depthmap'],
	 		'heatmap' : feature_dict['heatmap'],
			# 'pred_cam_trans' : pred_trans,
			# 'pred_cam_rot' : pred_rot
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

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
							   bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion,
								  momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

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


class PoseResNet(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		self.inplanes = 64
		self.deconv1_inplanes = None
		self.deconv2_inplanes = None
		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		super(PoseResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.deconv_layer1 = self._make_deconv_layer2(2048,1024)
		self.deconv_layer2 = self._make_deconv_layer2(1024,512)
		self.deconv_layer3 = self._make_deconv_layer2(512,128)
		self.deconv_layer4 = self._make_deconv_layer2(128,64)
		self.deconv_layer5 = self._make_deconv_layer2(64,1)
		
		self.final_layer =  nn.Sequential(
			nn.Conv2d(
			in_channels=512,
			out_channels=24,
			kernel_size=3,
			stride=1,
			padding=1
			),
			nn.BatchNorm2d(24, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True)
		)

	def _make_deconv_layer2(self,in_planes, out_planes, kernel=4, stride=2, padding=1):
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
	
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def _get_deconv_cfg(self, deconv_kernel, index):
		if deconv_kernel == 4:
			padding = 1
			output_padding = 0
		elif deconv_kernel == 3:
			padding = 1
			output_padding = 1
		elif deconv_kernel == 2:
			padding = 0
			output_padding = 0

		return deconv_kernel, padding, output_padding

	def _make_deconv_layer(self, num_layers, num_filters, num_kernels, init_inplanes):
		assert num_layers == len(num_filters), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'
		assert num_layers == len(num_kernels), \
			'ERROR: num_deconv_layers is different len(num_deconv_filters)'

		layers = []
		for i in range(num_layers):
			# 4, 1, 0
			kernel, padding, output_padding = \
				self._get_deconv_cfg(num_kernels[i], i)
			
			planes = num_filters[i] # [256,128,64,32,3]
			layers.append(
				nn.ConvTranspose2d(
					in_channels=init_inplanes,
					out_channels=planes,
					kernel_size=kernel,
					stride=2,
					padding=padding,
					output_padding=output_padding,
					bias=self.deconv_with_bias))
			layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
			layers.append(nn.ReLU(inplace=True))
			init_inplanes = planes

		return nn.Sequential(*layers)

	def forward(self, x): # -> 3x512x512
		
		x = self.conv1(x) # -> 64x256x256
		x_ = x
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x) # -> 64x128x128

		x_0 = self.layer1(x) # -> 256x128x128
		x_1 = self.layer2(x_0) # -> 512x64x64
		x_2 = self.layer3(x_1) # -> 1024x32x32
		temp_x = self.layer4(x_2) # -> 2048x16x16


		depthmap = self.deconv_layer1(temp_x) # -> 1024x32x32
		depthmap = self.deconv_layer2(depthmap+x_2) # -> 512x64x64
		depthmap = self.deconv_layer3(depthmap+x_1) # -> 128x128x128
		depthmap = self.deconv_layer4(depthmap) # -> 64x256x256
		depthmap = self.deconv_layer5(depthmap) # -> 1x512x512


		heatmap = self.deconv_layer1(temp_x) # -> 1024x32x32
		heatmap = self.deconv_layer2(heatmap+x_2) # -> 512x64x64

		heatmap = self.final_layer(heatmap) # -> 24x64x64

		res = {
			'depthmap':depthmap, 
			'heatmap':heatmap
		}
		return res

	def init_weights(self, pretrained=''):
		if os.path.isfile(pretrained):
			logger.info('=> init deconv weights from normal distribution')
			for name, m in self.deconv_layers.named_modules():
				if isinstance(m, nn.ConvTranspose2d):
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					if self.deconv_with_bias:
						nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.BatchNorm2d):
					logger.info('=> init {}.weight as 1'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			logger.info('=> init final conv weights from normal distribution')
			for m in self.final_layer.modules():
				if isinstance(m, nn.Conv2d):
					# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
					logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
					logger.info('=> init {}.bias as 0'.format(name))
					nn.init.normal_(m.weight, std=0.001)
					nn.init.constant_(m.bias, 0)

			# pretrained_state_dict = torch.load(pretrained)
			logger.info('=> loading pretrained model {}'.format(pretrained))
			# self.load_state_dict(pretrained_state_dict, strict=False)
			checkpoint = torch.load(pretrained)
			if isinstance(checkpoint, OrderedDict):
				state_dict = checkpoint
			elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
				state_dict_old = checkpoint['state_dict']
				state_dict = OrderedDict()
				# delete 'module.' because it is saved from DataParallel module
				for key in state_dict_old.keys():
					if key.startswith('module.'):
						# state_dict[key[7:]] = state_dict[key]
						# state_dict.pop(key)
						state_dict[key[7:]] = state_dict_old[key]
					else:
						state_dict[key] = state_dict_old[key]
			else:
				raise RuntimeError(
					'No state_dict found in checkpoint file {}'.format(pretrained))
			self.load_state_dict(state_dict, strict=False)
		else:
			logger.error('=> imagenet pretrained model dose not exist')
			logger.error('=> please download it first')
			raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
			   34: (BasicBlock, [3, 4, 6, 3]),
			   50: (Bottleneck, [3, 4, 6, 3]),
			   101: (Bottleneck, [3, 4, 23, 3]),
			   152: (Bottleneck, [3, 8, 36, 3])}

def get_pose_net(is_train, cfg=cfg,  **kwargs):
	num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
	# style = cfg.MODEL.STYLE

	block_class, layers = resnet_spec[num_layers]

	# if style == 'caffe':
	#     block_class = Bottleneck_CAFFE

	model = PoseResNet(block_class, layers, cfg, **kwargs)

	# if is_train and cfg.MODEL.INIT_WEIGHTS:
	# 	model.init_weights(cfg.MODEL.PRETRAINED)

	return model

class Regressor2(nn.Module):
	def __init__(self, feat_dim, smpl_mean_params):
		super().__init__()

		npose = 24 * 6
		self.procrustes={
			'rotation': torch.tensor([[ 0.9999647 , -0.00426632,  0.00723915],
										[ 0.00411011,  0.99976134,  0.02145697],
										[-0.00732896, -0.02142646,  0.9997435 ]]).T,
			'scale': 92.3,
			'translation': torch.tensor([  0.9100, 115.3339,  -2.7504])
		}
		self.flatten = nn.Flatten()
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.conv_block0 = self._make_block(1,24)
		self.conv_block1 = self._make_block(24,64) #512 512->256 256
		self.conv_block2 = self._make_block(64,128) #256 256->128 128
		self.conv_block3 = self._make_block(128,256) #128 128->64 64
		self.conv_block4 = self._make_block(256,512) #64 64->32 32
		# self.conv_block5 = self._make_block(512,1024) #32 32->16 16
		# self.conv_block6 = self._make_block(1024,2048) #16 16->4 4 : 32768
		
		self.downsample_heat = self._make_downsample(24,512,4,20,0)
		self.downsample_depth = self._make_downsample(1,512,17,33,0)

		self.fc1 = nn.Linear(feat_dim, 1024)
		self.drop1 = nn.Dropout()
		self.fc2 = nn.Linear(1024, 1024)
		self.drop2 = nn.Dropout()
		self.decpose = nn.Linear(1024, npose)
		self.decshape = nn.Linear(1024, 10)
		self.deccam_trans = nn.Linear(1024, 3)
		self.deccam_rot = nn.Linear(1024, 3)
		nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
		nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_trans.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_rot.weight, gain=0.01)

		self.smpl = SMPL(
			SMPL_MODEL_DIR,
			batch_size=64,
			create_transl=False
		)

		# mean_params = np.load(smpl_mean_params)
		# init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
		# init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
		# init_cam_trans = torch.from_numpy(mean_params['cam']).unsqueeze(0)
		# init_cam_rot = torch.from_numpy(mean_params['cam']).unsqueeze(0)
		# self.register_buffer('init_pose', init_pose)
		# self.register_buffer('init_shape', init_shape)
		# self.register_buffer('init_cam_trans', init_cam_trans)
		# self.register_buffer('init_cam_rot', init_cam_rot)

	def _make_downsample(self, in_channels, out_channels, kernel, stride, padding):
		return nn.Sequential(
			nn.Conv2d(in_channels=in_channels,
					  out_channels=out_channels,
					  kernel_size=kernel,
					  stride=stride,
					  padding=padding
					  ),
			nn.BatchNorm2d(out_channels,momentum=BN_MOMENTUM),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, heatmap, depthmap):
		x=heatmap # 24 64 64
		x2=depthmap # 1 512 512
		# TODO : heatmap 입력 argmax로 차원당 1 하나씩
		batch_size = x.shape[0]

		max_values, max_indices = torch.max(heatmap.view(batch_size,24,-1), dim=2)
		y_coords = max_indices // 64
		x_coords = max_indices % 64


		coords = torch.stack((x_coords, y_coords), dim=2)
		coords = self.flatten(coords) # 10x24x2


		residual_x = x # 24x64x64
		residual_x2 = x2
	

		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)
		x = self.conv_block4(x)
		residual_x = self.downsample_heat(residual_x)
		x = self.avgpool(x) #512

		x2 = self.conv_block0(x2) #->256
		x2 = self.conv_block1(x2) #->128
		x2 = self.conv_block2(x2) #->64
		x2 = self.conv_block3(x2) #->32
		x2 = self.conv_block4(x2)
		residual_x2 = self.downsample_depth(residual_x2)
		x2 = self.avgpool(x2+residual_x2) # 512

		x = x.squeeze()
		x2 = x2.squeeze()

		# for i in range(n_iter):
		xc = torch.cat([x, x2, coords], 1)
		xc = self.fc1(xc)
		xc = self.drop1(xc)
		xc = self.fc2(xc)
		xc = self.drop2(xc)
		pred_pose = self.decpose(xc)
		pred_shape = self.decshape(xc)
		pred_trans = self.deccam_trans(xc)
		pred_rot = self.deccam_rot(xc) 

		pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

		pred_output = self.smpl(
			betas=pred_shape,
			body_pose=pred_rotmat[:, 1:],
			global_orient=pred_rotmat[:, 0].unsqueeze(1),
			pose2rot=False
		)

		pred_vertices = pred_output.vertices
		pred_joints = pred_output.joints[:,:24,:]

		pred_joints = torch.einsum('bij,kj->bik',pred_joints,self.procrustes['rotation'].to(pred_joints.device))
		pred_joints *= self.procrustes['scale']
		pred_joints += self.procrustes['translation'].to(pred_joints.device)
		# pred_smpl_joints = pred_output.smpl_joints
		# pred_keypoints_2d = projection(pred_joints, pred_trans)

		pred_trans *= torch.tensor([29.0733, 12.2508, 55.9875]).to(pred_joints.device)
		pred_trans += torch.tensor([-5.2447, 141.3381, 33.3118]).to(pred_joints.device)
		# pred_keypoints_2d = self.fisheye_projection(pred_joints, pred_rot, pred_trans)
		pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

		res = {
			'theta'  : torch.cat([pred_trans, pred_shape, pose], dim=1),
			'verts'  : pred_vertices,
			# 'fisheye_kp_2d'  : pred_keypoints_2d,
			'kp_3d'  : pred_joints,
			# 'smpl_kp_3d' : pred_smpl_joints,
			'rotmat' : pred_rotmat,
			'pred_trans' : pred_trans,
			'pred_rot' : pred_rot,
			# 'pred_shape': pred_shape,
			# 'pred_pose': pred_pose,
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
			nn.LeakyReLU(inplace=True),
			nn.Dropout(p=0.2),
		)

class FisheyeProjection(nn.Module):
	def __init__(self):
		super().__init__()
		# TODO : K 스케일 확인할것 
		self.K = torch.tensor([[352.59619801644876, 0.0, 0.0],
				  [0.0, 352.70276325061578, 0.0],
				  [654.6810228318458, 400.952228031277, 1.0]]).T
		self.D = torch.tensor([-0.05631891929412012, -0.0038333424842925286,
						-0.00024681888617308917, -0.00012153386798050158])
		self.Mmaya = torch.tensor([[1, 0, 0, 0],
							[0, -1, 0, 0],
							[0, 0, -1, 0],
							[0, 0, 0, 1]],dtype=torch.float64)
	 

	def forward(self, points_3d, rotate, translation, resize=(512,512)):
		batch_size = len(points_3d)
		rotate = create_euler(rotate)
		rotate[:,:,:3,3] = translation.unsqueeze(1)
		Mf = torch.linalg.inv(rotate)
		M = self.Mmaya.T.float()@Mf.float()
		M = M.squeeze(1).to(points_3d.device)
		self.K = self.K.to(points_3d.device)
		# Transform points to camera coordinate system
		points_homogeneous = torch.cat((points_3d, torch.ones((batch_size,24,1), device=points_3d.device)), dim=-1)
		# P_ext = torch.cat((self.M[:3,:3], self.t), dim=-1)
		points_camera = torch.einsum('bij,bkj->bki',M,points_homogeneous)

		# Project points to the normalized image plane
		points_normalized = points_camera[:, :, :2] / points_camera[:, :, 2].unsqueeze(-1)

		# Apply fisheye distortion
		r = torch.norm(points_normalized, p=2, dim=-1)
		theta = torch.atan2(r, torch.ones_like(r))
		theta_d = theta * (1 + self.D[0] * theta**2 + self.D[1] * theta**4 + self.D[2] * theta**6 + self.D[3] * theta**8)
		points_distorted = points_normalized * (theta_d / r).unsqueeze(-1)

		# Project distorted points to the image plane
		res = torch.matmul(points_distorted, self.K[:2, :2]) + self.K[:2, 2]
		if resize is not None:
			H,W=800,1280
			norm_res = res/torch.tensor([W,H]).to(res.device)
			res = norm_res * torch.tensor(resize).to(res.device)
		return res