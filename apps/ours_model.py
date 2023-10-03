import torch.nn as nn
import torch.nn.functional as F
import torch
from ours_backbone import ResNet50
from ours_WASP import WASP
from ours_decoder import Decoder
from ours_regressor import Regressor

class Ours_model(nn.Module):
	def __init__(self, pretrained_path:str =None) -> None:
		super().__init__()
		self.heatmap_backbone = ResNet50()
		# self.heatmap_WASP = WASP()
		self.with_out_wasp_h = nn.Conv2d(2048,256,3)
		self.heatmap_decoder = Decoder('heat')
		
		self.depthmap_backbone = ResNet50()
		# self.depthmap_WASP = WASP()
		self.with_out_wasp_d = nn.Conv2d(2048,256,3)
		
		self.depthmap_decoder = Decoder('depth')
		
		self.deconv_depth = nn.Sequential(
			nn.ConvTranspose2d(32, 1,  kernel_size=4, stride=2, padding=1),
			nn.Sigmoid(),
		)
		self.deconv_sil = nn.ConvTranspose2d(32, 3,  kernel_size=4, stride=2, padding=1)
		
		self.seg_decoder = Decoder('depth')
		self.with_out_wasp_s = nn.Conv2d(2048,256,3)

		
		self.Regressor = Regressor(256,256)
		if pretrained_path:
			tempmodel_ckpt = torch.load(pretrained_path)
			self.load_state_dict(tempmodel_ckpt)


	def forward(self,x):

		input_ = x['image']
		heat_x,heat_low = self.heatmap_backbone(input_)
		# heat_x = self.heatmap_WASP(heat_x)
		heat_x = self.with_out_wasp_h(heat_x)
		heat_embed = heat_x
		heat_x = self.heatmap_decoder(heat_x,heat_low)
		heatmap = F.interpolate(heat_x, size=(64,64), mode='bilinear', align_corners=True)

		depth_x,depth_low = self.depthmap_backbone(input_)
		# depth_x = self.depthmap_WASP(depth_x)
		seg_x = self.with_out_wasp_s(depth_x)
		depth_x = self.with_out_wasp_d(depth_x)
		seg_embed = seg_x
		depth_embed = depth_x
		seg_x = self.seg_decoder(seg_x,depth_low)
		depth_x = self.depthmap_decoder(depth_x,depth_low)
		
		depthmap = self.deconv_depth(depth_x)
		silhouette = self.deconv_sil(seg_x)

		depthmap = F.interpolate(depthmap, size=(256,256), mode='bilinear', align_corners=True)
		silhouette = F.interpolate(silhouette, size=(256,256), mode='bilinear', align_corners=True)

		reg_dict = self.Regressor(
			heatmap_embed = heat_embed, 
			heatmap = None, 
			depthmap_embed = depth_embed, 
			depthmap = depthmap,
			seg_embed = seg_embed,
		)

		res ={
			'regressor_dict' : reg_dict,
			'heatmap' : heatmap,
			'depthmap' : depthmap,
			'silhouette' : silhouette,
			'pred_pose' : reg_dict['pred_joint']
		}
		return res