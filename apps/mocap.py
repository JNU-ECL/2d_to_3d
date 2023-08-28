# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.

@author: Denis Tome'

"""
import cv2
import os
import torch
from PIL import Image
from skimage import io as sio
import numpy as np
from base import BaseDataset
from utils import io, config, transform
from torchvision import transforms


class Mocap(BaseDataset):
	"""Mocap Dataset loader"""

	ROOT_DIRS = ['rgba','depth','json','objectId']
	CM_TO_M = 100

	def index_db(self):

		return self._index_dir(self.path)

	def _index_dir(self, path):
		"""Recursively add paths to the set of
		indexed files

		Arguments:
			path {str} -- folder path

		Returns:
			dict -- indexed files per root dir
		"""

		indexed_paths = dict()
		sub_dirs, _ = io.get_subdirs(path)
		if set(self.ROOT_DIRS) <= set(sub_dirs):

			# get files from subdirs
			n_frames = -1

			# let's extract the rgba and json data per frame
			for sub_dir in self.ROOT_DIRS:
				d_path = os.path.join(path, sub_dir)
				_, paths = io.get_files(d_path)

				if n_frames < 0:
					n_frames = len(paths)
				else:
					if len(paths) != n_frames:
						self.logger.error(
							'Frames info in {} not matching other passes'.format(d_path))

				encoded = [p.encode('utf8') for p in paths]
				indexed_paths.update({sub_dir: encoded})

			return indexed_paths

		# initialize indexed_paths
		for sub_dir in self.ROOT_DIRS:
			indexed_paths.update({sub_dir: []})

		# check subdirs of path and merge info
		for sub_dir in sub_dirs:
			indexed = self._index_dir(os.path.join(path, sub_dir))

			for r_dir in self.ROOT_DIRS:
				indexed_paths[r_dir].extend(indexed[r_dir])

		return indexed_paths

	def _process_points(self, data):
		"""Filter joints to select only a sub-set for
		training/evaluation

		Arguments:
			data {dict} -- data dictionary with frame info

		Returns:
			np.ndarray -- 2D joint positions, format (J x 2)
			np.ndarray -- 3D joint positions, format (J x 3)
		"""

		p2d_orig = np.array(data['pts2d_fisheye']).T
		p3d_orig = np.array(data['pts3d_fisheye']).T
		joint_names = {j['name'].replace('mixamorig:', ''): jid
					   for jid, j in enumerate(data['joints'])}

		# ------------------- Filter joints -------------------

		p2d = np.empty([len(config.skel_15), 2], dtype=p2d_orig.dtype)
		p3d = np.empty([len(config.skel_16), 3], dtype=p2d_orig.dtype)
		p3d_p = np.empty([len(config.skel_16), 3], dtype=p2d_orig.dtype)

		for jid, j in enumerate(config.skel_15.keys()):
			p2d[jid] = p2d_orig[joint_names[j]]
			
		for jid, j in enumerate(config.skel_16.keys()):	
			p3d[jid] = p3d_orig[joint_names[j]]

		for jid, j_ in enumerate(config.skel_16.values()):
			if j_['parent']:
				p3d_p[jid] = p3d_orig[joint_names[j_['parent']]] / self.CM_TO_M
				

		p3d /= self.CM_TO_M
		Neck = p3d_orig[joint_names['Neck']] / self.CM_TO_M
		return p2d, p3d, Neck, p3d_p

	def _get_image(self,img_path,resize=(256,256)):
		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		transform_ = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(resize,Image.BILINEAR),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		return transform_(img)
	
	def _get_joint3d(self,data,Neck):
		p3d = data
		joint_zeroed = Neck[np.newaxis]
		# update p3d
		p3d -= joint_zeroed
		return torch.from_numpy(p3d).float()
	
	def _get_joint2d(self,data,resize=(256,256)):
		p2d = data
		p2d = torch.from_numpy(p2d).float() # 0~1280, 0~800
		H,W=800,1280
		norm_res = p2d/torch.tensor([W,H])
		p2d = norm_res * torch.tensor(resize)
		return p2d
	

	# 가우시안 분포를 생성하는 함수
	def generate_gaussian_heatmap(self,joint_location,image_size=None, sigma=2):
		x, y = joint_location
		x, y = max(np.array(1),np.array(x)),max(np.array(1),np.array(y))
		grid_y, grid_x = np.mgrid[0:image_size[1], 0:image_size[0]]
		dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
		heatmap = np.exp(-dist / (2 * sigma**2))
		return heatmap
	
	# heatmap GT 생성 함수
	def generate_heatmap_gt(self, joint_location, image_size=None,sigma=2):
		heatmap_gt = np.zeros((len(joint_location),image_size[1], image_size[0]), dtype=np.float32)
		for i, joint in enumerate(joint_location):
			heatmap_gt[i, :, :] = self.generate_gaussian_heatmap(joint, image_size, sigma)
		return heatmap_gt

	def _get_heatmap(self,data,sigma=2,resize=(32,32)):
		res=None
		fisheye_joint_labels = self._get_joint2d(data,resize=resize)
		res = self.generate_heatmap_gt(fisheye_joint_labels,sigma=sigma,image_size=resize)
		return res

	def _get_depth(self,depth_path,resize=(256,256)):
		img = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(resize, Image.BILINEAR),
			transforms.ToTensor(),
			# transforms.Normalize((0.5), (0.5)),
		])
		return transform(img)
	
	def _get_cam(self,data):
		trans = torch.tensor(data['camera']['trans'])
		rot = torch.tensor(data['camera']['rot']) * np.pi / 180.0
		return trans, rot

	def _get_silhouette(self,img_path,resize=(256,256)):
		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		silhouette = np.zeros_like(img[:,:,0])
	
		not_black = np.any(img != [0, 0, 0], axis=-1)

		silhouette[not_black] = 255

		# h,w,c = img.shape
		# c_h,c_w = h//2,w//2
		# radius = min(c_h,c_w) + 100
		# mask = np.zeros_like(img)
		# cv2.circle(mask, (c_w,c_h), radius, (255, 255, 255), -1)
		# cropped_img = cv2.bitwise_and(img, mask)
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(resize, Image.BILINEAR),
			transforms.ToTensor(),
		])
		return transform(silhouette)

	def _get_normal(self,p3d_,p3d_p,Neck):
		
		
		joint_zeroed = Neck[np.newaxis]
		# update p3d

		p3d_p -= joint_zeroed
		p3d_ -= joint_zeroed
		
		normal = p3d_ - p3d_p
		
		return torch.from_numpy(normal).float()
	
	def _get_image_zoom(self,img_path,resize=(512,512),crop_size=(256,256)):
		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		transform_ = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(resize,Image.BILINEAR),
			transforms.CenterCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		return transform_(img)

	def __getitem__(self, index):

		# load image
		img_path = self.index['rgba'][index].decode('utf8')
		# img = sio.imread(img_path).astype(np.float32)
		# img /= 255.0

		# read joint positions
		json_path = self.index['json'][index].decode('utf8')
		data = io.read_json(json_path)
		p2d_, p3d_, Neck, p3d_p = self._process_points(data) 

		depth_path = self.index['depth'][index].decode('utf8')

		obj_path = self.index['objectId'][index].decode('utf8')

		# get action name
		action = data['action']
		img = self._get_image(img_path)
		p3d = self._get_joint3d(p3d_,Neck)
		p2d = self._get_joint2d(p2d_)
		depth = self._get_depth(depth_path)
		heatmap = self._get_heatmap(p2d_)
		cam = self._get_cam(data)
		silhouette = self._get_silhouette(obj_path)
		normal = self._get_normal(p3d_,p3d_p,Neck)
		img_zoom = self._get_image_zoom(img_path)
		return {
			'info' : img_path,
			'camera_info' : cam,
			'image' : img, 
			'fisheye_joints_2d' : p2d, 
			'joints_3d_cam' : p3d,
			'depth' : depth,
			'heatmap' : heatmap, 
			'action' : action,
			'silhouette' : silhouette,
			'normal' : normal,
			'image_zoom' : img_zoom,
		}

	def __len__(self):
		return len(self.index[self.ROOT_DIRS[0]])
