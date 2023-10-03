import math
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


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


		self.fc1 = nn.Linear(heatmapfeat_c + depthmapfeat_c+256+128, 1024)
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

		
	def forward(self, heatmap_embed=None, heatmap=None, depthmap_embed=None, depthmap=None, seg_embed=None):
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
		if seg_embed is not None:
			x5 = self.avgpool(seg_embed)
			seg_e_feat=x5.view(batch_size,-1)

		heatmap_e_feat = x.view(batch_size,-1)
		depth_e_feat = x2.view(batch_size,-1)
		depthmap_feat = x3.view(batch_size,-1) 
		# heatmap_feat = x4.view(batch_size,-1)

		total_feat = torch.cat([heatmap_e_feat, seg_e_feat, depth_e_feat, depthmap_feat], 1)
		# total_feat = torch.cat([heatmap_e_feat], 1)
		
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