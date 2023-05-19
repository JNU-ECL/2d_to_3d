
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
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
	SoftSilhouetteShader
)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


SMPL_MEAN_PARAMS = '/workspace/2d_to_3d/model/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/workspace/2d_to_3d/model/smpl'
use_cuda = torch.cuda.is_available()


class TempModel(nn.Module):
	def __init__(self,pretrained_path:str =None):
		super().__init__()
		only_resnet=False
		self.heatmap_module = get_pose_net('heatmap')
		self.depthmap_module = get_pose_net('depth')
		self.regressor=Regressor(depthmapfeat_c=512,heatmapfeat_c=512)
		
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			if only_resnet:
				feature_model1_state_dict = OrderedDict()
				for k,v in tempmodel_ckpt.items():
					if k.startswith('feature_model1.'):
						new_key = k[len('feature_model1.'):]
						feature_model1_state_dict[new_key] = v
				self.feature_model1.load_state_dict(feature_model1_state_dict)
			else:
				self.load_state_dict(tempmodel_ckpt)

	def forward(self, x):
		res = {}
		image_ = x['image']
		
		heatmap_feat=self.heatmap_module(image_)
		depth_feat=self.depthmap_module(image_)

		regressor_res_dict=self.regressor(
			heatmap_feat['embed_feature'].detach(),
			depth_feat['embed_feature'].detach(),
		)

	

		res.update({
   	  		'regressor_dict' : regressor_res_dict,
	 		'depthmap' : depth_feat['depthmap'],
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
	
class PoseResNet_depth(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		self.inplanes = 64

		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		super(PoseResNet_depth, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# self.deconv_layer1 = self._make_deconv_layer(2048,1024)
		# self.deconv_layer2 = self._make_deconv_layer(1024,512)
		# self.deconv_layer3 = self._make_deconv_layer(512,256)
		# self.deconv_layer4 = self._make_deconv_layer(256,128)
		# self.deconv_layer5 = self._make_deconv_layer(128,1)

		self.deconv_layer1 = self._make_deconv_layer(512,256)
		self.deconv_layer2 = self._make_deconv_layer(256,128)
		self.deconv_layer3 = self._make_deconv_layer(128,64)
		self.deconv_layer4 = self._make_deconv_layer(64,32)
		self.deconv_layer5 = self._make_deconv_layer(32,1)
		

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


	def forward(self, x): # -> 3x256x256
		# batch_size=len(x)
		x = self.conv1(x) # -> 64x128x128
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x) # -> 64x64x64

		x_0 = self.layer1(x) # -> 64x64x64
		x_1 = self.layer2(x_0) # -> 128x32x32
		x_2 = self.layer3(x_1) # -> 256x16x16
		temp_x = self.layer4(x_2) # -> 512x8x8

		depthmap = self.deconv_layer1(temp_x) # -> 256x16x16
		depthmap = self.deconv_layer2(depthmap+x_2) # -> 128x32x32
		depthmap = self.deconv_layer3(depthmap+x_1) # -> 64x64x64
		depthmap = self.deconv_layer4(depthmap+x_0) # -> 32x128x128
		depthmap = self.deconv_layer5(depthmap) # -> 1x256x256

		res = {
			'depthmap':depthmap, 
			'embed_feature':temp_x,
		}

		return res

class PoseResNet_heatmap(nn.Module):

	def __init__(self, block, layers, cfg, **kwargs):
		self.inplanes = 64
		extra = cfg.MODEL.EXTRA
		self.deconv_with_bias = extra.DECONV_WITH_BIAS

		super(PoseResNet_heatmap, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		# self.deconv_layer1_ = self._make_deconv_layer(2048,1024)
		# self.deconv_layer2_ = self._make_deconv_layer(1024,512)
		# self.deconv_layer3_ = self._make_deconv_layer(512,256)
		
		self.deconv_layer1_ = self._make_deconv_layer(512,256)
		self.deconv_layer2_ = self._make_deconv_layer(256,128)
		self.deconv_layer3_ = self._make_deconv_layer(128,64)

		
		self.final_layer =  nn.Sequential(
			nn.Conv2d(
			in_channels=64,
			out_channels=15,
			kernel_size=3,
			stride=1,
			padding=1
			),
			nn.BatchNorm2d(15, momentum=BN_MOMENTUM),
			nn.ReLU(inplace=True)
		)

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

	def forward(self, x): # -> 3x256x256
		# batch_size=len(x)
		x = self.conv1(x) # -> 64x128x128
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x) # -> 64x64x64

		x_0 = self.layer1(x) # -> 64x64x64     //256x64x64
		x_1 = self.layer2(x_0) # -> 128x32x32  //512x32x32
		x_2 = self.layer3(x_1) # -> 256x16x16  //1024x16x16
		temp_x = self.layer4(x_2) # -> 512x8x8 // 2048x8x8

		heatmap_1 = self.deconv_layer1_(temp_x) # -> 256x16x16 # change all down
		heatmap_2 = self.deconv_layer2_(heatmap_1+x_2) # -> 128x32x32
		heatmap_3 = self.deconv_layer3_(heatmap_2+x_1) # -> 64x64x64
	
		heatmap_4 = self.final_layer(heatmap_3) # -> 15x64x64

		res = {
			'heatmap':heatmap_4,
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
	
	block_class, layers = resnet_spec[34]

	# if style == 'caffe':
	#     block_class = Bottleneck_CAFFE
	if model_n == 'heatmap':
		model = PoseResNet_heatmap(block_class, layers, cfg, **kwargs)
	elif model_n == 'depth':
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


		self.fc1 = nn.Linear(heatmapfeat_c + depthmapfeat_c, 1024)
		self.bn1 = nn.BatchNorm1d(1024,momentum=BN_MOMENTUM)
		self.relu1 = nn.ReLU()
		self.drop1 = nn.Dropout()


		self.decjoint = nn.Linear(1024, njoint)
		self.deccam_trans = nn.Linear(1024, 3)
		self.deccam_rot = nn.Linear(1024, 3)

		nn.init.xavier_uniform_(self.decjoint.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_trans.weight, gain=0.01)
		nn.init.xavier_uniform_(self.deccam_rot.weight, gain=0.01)

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

		
	def forward(self, heatmap_embed, depthmap_embed):
		x=heatmap_embed # 512x8x8
		x2=depthmap_embed # 512x8x8
		# TODO : heatmap 입력 argmax로 차원당 1 하나씩
		batch_size = x.shape[0]

		x = self.avgpool(x) #512

		x2 = self.avgpool(x2) # 512

		heatmap_feat = x.squeeze()
		depth_feat = x2.squeeze()


		total_feat = torch.cat([heatmap_feat, depth_feat], 1)

		pred_joint = self.fc1(total_feat)
		pred_joint = self.bn1(pred_joint)
		pred_joint = self.relu1(pred_joint)
		pred_joint = self.drop1(pred_joint)
		pose_residual = pred_joint

		for layer in self.bilinear_layer_pose:
			pred_joint = layer(pred_joint)
			pred_joint += pose_residual


		pred_joint = self.decjoint(pred_joint).view(-1,16,3)
		# pred_trans = self.deccam_trans(total_feat)
		# pred_rot = self.deccam_rot(total_feat) 
		
		# pred_trans *= torch.tensor([29.0733, 12.2508, 55.9875]).to(pred_trans.device)
		# pred_trans += torch.tensor([-5.2447, 141.3381, 33.3118]).to(pred_trans.device)

		"""
		pred_vertices = pred_output.vertices
		pred_joints = torch.concat((pred_output.joints[:,:13,:],
									pred_output.joints[:,16:25,:]),dim=1)
		pred_joints_raw = pred_output.joints[:,:25,:]

		

		world_coord_joints_raw = self._world_coord(pred_joints_raw)
		cam_coord_joints_raw = self._cam_coord(world_coord_joints_raw,pred_rot,pred_trans,batch_size)
		
		
		world_coord_joints = self._world_coord(pred_joints)
		cam_coord_joints = self._cam_coord(world_coord_joints,pred_rot,pred_trans,batch_size)
		pred_keypoints_2d = self.fisheye_projection(world_coord_joints, pred_rot, pred_trans)
		pred_heatmap = self.get_gaussian_heatmap(world_coord_joints, pred_rot, pred_trans)

		# pred_smpl_joints = pred_output.smpl_joints
		# pred_keypoints_2d = projection(pred_joints, pred_trans)
		
		# pred_keypoints_2d = self.fisheye_projection(pred_joints, pred_rot, pred_trans)
		world_coord_vertices = self._world_coord(pred_vertices)
		cam_coord_vertices = self._cam_coord(world_coord_vertices,pred_rot,pred_trans,batch_size)
		
		R = create_euler(pred_rot,True).view(batch_size,3,3)
		T = pred_trans.view(batch_size,3)
		cameras = OpenGLPerspectiveCameras(device=cam_coord_vertices.device, R=R, T=T)

		raster_settings = RasterizationSettings(
			image_size=512, 
			blur_radius=0.0, 
			faces_per_pixel=1, 
		)
		# TODO vertices to cam coord
		# Create a silhouette shader
		shader = SoftSilhouetteShader()

		renderer = MeshRenderer(
			rasterizer=MeshRasterizer(
				cameras=cameras, 
				raster_settings=raster_settings
			),
			shader=shader
		)
		faces = self.smpl_layer.th_faces
		faces = faces[None].expand(batch_size, -1, -1).to(cam_coord_vertices.device)
		meshes = Meshes(verts=cam_coord_vertices, faces=faces)
		silhouette = renderer(meshes)
		distortion_silhouette = self.fisheye_distortion(silhouette)
		

		pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

		"""
		res = {
			# 'pred_trans' : pred_trans,
			# 'pred_rot' : pred_rot,
			'pred_joint': pred_joint,
		}
		return res

	def generate_gaussian_heatmap(self, joint_location, image_size=(64, 64), sigma=1.8):
		epsilon = 1e-9
		x, y = joint_location
		# x, y = torch.tensor(x), torch.tensor(y)
		grid_y, grid_x = torch.meshgrid(torch.arange(0, image_size[1]), torch.arange(0, image_size[0]))
		grid_x = grid_x.to(x.device)
		grid_y = grid_y.to(y.device)
		dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
		heatmap = torch.exp(-dist / ((2 * sigma**2)+epsilon))
		return heatmap

	def generate_heatmap_gt(self, joint_location, image_size=(64, 64), sigma=1.8):
		batch_size,joint_num,_=joint_location.shape
		heatmap_gt = torch.zeros((batch_size, joint_num, image_size[1], image_size[0]), dtype=torch.float32, device=joint_location.device)
		for i in range(batch_size):	
			for j, joint in enumerate(joint_location[i]):
				heatmap_gt[i, j, :, :] = self.generate_gaussian_heatmap(joint, image_size, sigma)
		return heatmap_gt

	def get_gaussian_heatmap(self, world_coord_joints, pred_rot, pred_trans):
		res = None
		fisheye_joint_labels = self.fisheye_projection(world_coord_joints, pred_rot, pred_trans, resize=(64, 64))
		res = self.generate_heatmap_gt(fisheye_joint_labels)
		return res

	def _world_coord(self,pred_joints):
		pred_joints = torch.einsum('bij,kj->bik',pred_joints,self.procrustes['rotation'].to(pred_joints.device))
		pred_joints *= self.procrustes['scale']
		pred_joints += self.procrustes['translation'].to(pred_joints.device) # world coo
		return pred_joints

	def _cam_coord(self,pred_joints,pred_rot,pred_trans,batch_size):
		cam_intrinsic = create_euler(pred_rot).view(batch_size,4,4)
		cam_intrinsic[:,:3,3] = pred_trans
		cam_intrinsic = torch.linalg.inv(cam_intrinsic)
		cam_intrinsic = self.Mmaya@cam_intrinsic
		
		pred_joints_world = torch.cat((pred_joints,torch.ones((batch_size,len(pred_joints[1]),1),device=pred_joints.device)),dim=-1)
		pred_joints_cam = torch.einsum('bij,bkj->bik',pred_joints_world,cam_intrinsic.to(pred_joints_world.device))
		pred_joints_cam = pred_joints_cam[:,:,:3]
		return pred_joints_cam
	
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
		epsilon = 1e-9
		batch_size = len(points_3d)
		rotate = create_euler(rotate)
		rotate[:,:,:3,3] = translation.unsqueeze(1)
		Mf = torch.linalg.inv(rotate)
		M = self.Mmaya.T.float()@Mf.float()
		M = M.squeeze(1).to(points_3d.device)
		self.K = self.K.to(points_3d.device)
		# Transform points to camera coordinate system
		points_homogeneous = torch.cat((points_3d, torch.ones((batch_size,points_3d.shape[1],1), device=points_3d.device)), dim=-1)
		# P_ext = torch.cat((self.M[:3,:3], self.t), dim=-1)
		points_camera = torch.einsum('bij,bkj->bki',M,points_homogeneous)

		# Project points to the normalized image plane
		points_normalized = points_camera[:, :, :2] / (points_camera[:, :, 2].unsqueeze(-1) + epsilon)

		# Apply fisheye distortion
		
		r = torch.norm(points_normalized + epsilon, p=2, dim=-1)
		r = torch.clamp(r, min=epsilon)
		theta = torch.atan2(r, torch.ones_like(r) + epsilon)
		theta_d = theta * (1 + self.D[0] * theta**2 + self.D[1] * theta**4 + self.D[2] * theta**6 + self.D[3] * theta**8)
		points_distorted = points_normalized * (theta_d / r).unsqueeze(-1)

		# Project distorted points to the image plane
		res = torch.matmul(points_distorted, self.K[:2, :2]) + self.K[:2, 2]
		if resize is not None:
			H,W=800,1280
			norm_res = res/torch.tensor([W,H]).to(res.device)
			res = norm_res * torch.tensor(resize).to(res.device)
		return res

class FisheyeDistortion(nn.Module):
	def __init__(self):
		super().__init__()
		self.D = torch.tensor([-0.05631891929412012, -0.0038333424842925286,
			-0.00024681888617308917, -0.00012153386798050158])

	def forward(self, x):
		epsilon = 1e-9
		# Normalize coordinates to [-1, 1]
		x = (x - 0.5) * 2

		# Compute radial distortion
		r = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) + epsilon)
		theta = torch.atan2(r, torch.ones_like(r) + epsilon)
		theta_d = theta * (1 + self.D[0] * theta**2 + self.D[1] * theta**4 + self.D[2] * theta**6 + self.D[3] * theta**8)
		x_distorted = x * (theta_d / r)

		# Rescale to [0, 1]
		x_distorted = x_distorted / 2 + 0.5

		return x_distorted