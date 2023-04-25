import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from typing import Tuple, List
from torch.utils.data import Dataset, Subset, random_split
from PIL import Image, ImageOps
import os
import glob
from tqdm import tqdm
from geometry import rot6d_to_rotmat, projection_temp, rotation_matrix_to_angle_axis, projection_temp_dataset
from matplotlib import pyplot as plt
import transformations
import cv2
import platform
"""import custom_utils
from custom_utils import get_ocr, ocr_to_coco, coco_to_mask

import key

key_list = key.new_list"""

"""
TrainSet
├── female_001_a_a
│   ├── env 01
│   │   └── cam_down
│   │   	├── depth
│   │   	├── json
│   │   	├── objectId
│   │   	├── rgba
│   │   	├── rot
│   │   	└── worldp
│   ├── ...
│   └── env 03
└── ...
"""
split_token = '/' if 'Linux' in platform.platform() else '\\'
class BaseAugmentation:
	def __init__(self, resize=[512,512],  **args):
		self.transform = transforms.Compose([
			transforms.Resize(resize, Image.BILINEAR),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

	def __call__(self, image):
		return self.transform(image)

def load_data(dataroot,data_type)->dict:
	extension = '*.json' if data_type=='json' else '*.png'
	subjects=[]
	total_len=0
	res={}
	subjects={}
	for folder in sorted(glob.glob(os.path.join(dataroot,'*'))):
		subject_name=folder.split(split_token)[-1]
		path_dict={}
		for env in sorted(glob.glob(os.path.join(folder,'*'))):
			env_name=env.split(split_token)[-1]
			path_=sorted(glob.glob(os.path.join(env,'cam_down',data_type,extension)))
			total_len+=len(path_)
			path_dict.update({      
				env_name:path_
				})
		subjects.update({
			subject_name : path_dict
		})
	res.update({
		'subjects' : subjects,
		'total_len' : total_len,
	})
		
	return res


def get_path(index,datadict)->str:
	for sub in datadict['subjects'].keys():
		for env in datadict['subjects'][sub].keys():
			if index<len(datadict['subjects'][sub][env]):
				return datadict['subjects'][sub][env][index]
			index-=len(datadict['subjects'][sub][env])
	return -1


class temp_dataset(Dataset):
	def __init__(self,dataroot,mode = "train") -> None:
		self.mode = mode
		self.root = dataroot
		# self.load_size = loadsize    
		self.is_train = (mode=='train')
		self.val_ratio=0.2
		
		self.IMAGE = load_data(self.root,'rgba')
		self.subject_names=self.IMAGE.keys()

		self.xR_2_SMPL=[2,31,61,62,27,57,63,4,34,64,29,59,0,28,58,1,3,33,5,35,6,36,11,41]
		self.skip_num = []

		self.Khmc = torch.tensor([[352.59619801644876, 0.0, 0.0],
					[0.0, 352.70276325061578, 0.0],
					[654.6810228318458, 400.952228031277, 1.0]]).T
		self.kd = torch.tensor([-0.05631891929412012, -0.0038333424842925286,
						-0.00024681888617308917, -0.00012153386798050158])

		self.Mmaya = torch.tensor([[1., 0., 0., 0.],
							[0., -1., 0., 0.],
							[0., 0., -1., 0.],
							[0., 0., 0., 1.]])
		if self.is_train:
			self.JSON = load_data(self.root,'json')
			self.SEGMAP = load_data(self.root,'objectId')
			self.DEPTH = load_data(self.root,'depth')
			self.SMPL = os.path.join(self.root,'smpl')

		# # PIL to tensor
		# self.transform = transforms.Compose([
		# 	transforms.Resize([512,512], Image.BILINEAR),
		# 	transforms.ToTensor(),
		# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		# ])

		# # augmentation
		# self.aug_trans = transforms.Compose([
		#     transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
		#                            hue=opt.aug_hue)
		# ])

	# def set_transform(self, transform):
	# 	self.transform = transform  
			
	def get_rgba(self,img_path,resize=(512,512)):
		img = cv2.imread(img_path,cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
			# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])
		return transform(img)
		
	def get_camera(self,json_path):
		temp_json=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())
		trans=torch.tensor(temp_json['camera']['trans']) 
		# trans -= torch.tensor([-5.2447, 141.3381, 33.3118])
		# trans /= torch.tensor([29.0733, 12.2508, 55.9875])
		rot=torch.tensor(temp_json['camera']['rot']) * np.pi / 180.0
		return trans,rot

	def get_depth(self,depth_path,resize=(512,512)):
		img = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
		# h,w = img.shape
		# c_h,c_w = h//2,w//2
		# radius = min(c_h,c_w) + 100
		# mask = np.zeros_like(img)
		# cv2.circle(mask, (c_w,c_h), radius, (255, 255, 255), -1)
		# cropped_img = cv2.bitwise_and(img, mask)
		
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(resize, Image.BILINEAR),
			transforms.ToTensor(),
			# transforms.Normalize((0.5), (0.5)),
		])

		return transform(img)

	def get_3d_joints(self,json_path,cam_coord=False):
		"""
		This function get xR_ego_dataset and convert to SMPL style points.
		Input:
			json_path : ego_dataset path
		Output:
			tensor (24,3) : SMPL_style_joints
		"""
		temp_json=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())
		if cam_coord:
			joints = temp_json['pts3d_fisheye']
			assert len(joints) == 3, f'pts3d_fisheye None{json_path}'
		else:
			joints = np.vstack([j['trans'] for j in temp_json['joints']]).T
		x=torch.tensor(joints[0])
		y=torch.tensor(joints[1])
		z=torch.tensor(joints[2])
		temp_joints = [x,y,z]
		temp_joints = torch.stack(temp_joints,0).to(torch.float32).T
		ego_pose_label = temp_joints
		
		SMPL_label=[]
		for xR_idx in self.xR_2_SMPL:
			if xR_idx in self.skip_num : continue
			SMPL_label.append(temp_joints[xR_idx])
		SMPL_label=torch.stack(SMPL_label)
		# self.temp_view(SMPL_label)
		return SMPL_label
	#TODO : 굳이 SMPL 에서 2d projection을 할 필요는 없지않나.. 	
	# def get_2d_joints(self,json_path):
	# 	temp_json=None
	# 	res=None
	# 	with open(json_path,'r') as f:
	# 		temp_json=json.loads(f.read())
	# 	h_fov = torch.tensor(temp_json['camera']['cam_fov'])
	# 	translation = torch.tensor(temp_json['camera']['trans'])
	# 	rotation = torch.tensor(temp_json['camera']['rot']) * np.pi / 180.0
	# 	joints_3d = self.get_3d_joints(json_path)
		
		
	# 	joints_2d = projection_temp_dataset(joints_3d,translation,rotation)
	# 	return joints_2d
	
	def get_fisheye_2d_joints(self, json_path, resize=(512,512), cam_coord = True):
		"""
		TODO:
		Input:
			json_path : ego_dataset path
		Output:
			numpy array (24,2) : fisheye projection
		"""
		temp_join=None
		res=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())

		if cam_coord:
			temp_joints = torch.tensor(temp_json['pts2d_fisheye']).T # json 안쪽에 pts2d_fisheye 데이터 결측치 있음 
			SMPL_label=[]
			for xR_idx in self.xR_2_SMPL:
				if xR_idx in self.skip_num : continue
				SMPL_label.append(temp_joints[xR_idx])
			SMPL_label=torch.stack(SMPL_label)
			res = SMPL_label
			if resize is not None:
				H,W=800,1280
				norm_res = res/torch.tensor([W,H])
				res = norm_res * torch.tensor(resize)	
			return res

		translation = torch.tensor(temp_json['camera']['trans'])
		rotation = torch.tensor(temp_json['camera']['rot']) * np.pi / 180.0
		joints_3d = self.get_3d_joints(json_path).T
		
		
		Khmc = torch.tensor([[352.59619801644876, 0.0, 0.0],
				  [0.0, 352.70276325061578, 0.0],
				  [654.6810228318458, 400.952228031277, 1.0]]).T
		kd = torch.tensor([-0.05631891929412012, -0.0038333424842925286,
						-0.00024681888617308917, -0.00012153386798050158])

		Mmaya = torch.tensor([[1., 0., 0., 0.],
							[0., -1., 0., 0.],
							[0., 0., -1., 0.],
							[0., 0., 0., 1.]])
			
		Mf = transformations.euler_matrix(rotation[0],
										rotation[1],
										rotation[2],
										'sxyz')

		Mf[0:3, 3] = translation
		Mf = torch.linalg.inv(torch.tensor(Mf)).type(torch.FloatTensor)
		M = (Mmaya.T)@(Mf)


		Xj = M[0:3, 0:3]@(joints_3d) + M[0:3, 3:4]
		Xj = Xj.numpy()
		Khmc = Khmc.numpy()
		kd = kd.numpy()

		pts2d, jac = cv2.fisheye.projectPoints(
			Xj.T.reshape((1, -1, 3)),
			(0, 0, 0),
			(0, 0, 0),
			Khmc,
			kd
		)
		res = pts2d.squeeze(0)
		if resize is not None:
			H,W=800,1280
			norm_res = res/torch.tensor([W,H])
			res = norm_res * torch.tensor(resize)
		return res
	
	# 가우시안 분포를 생성하는 함수
	def generate_gaussian_heatmap(self,joint_location,image_size=[64,64], sigma=2):
		x, y = joint_location
		x, y = np.array(x),np.array(y)
		grid_y, grid_x = np.mgrid[0:image_size[1], 0:image_size[0]]
		dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
		heatmap = np.exp(-dist / (2 * sigma**2))
		return heatmap
	
	# heatmap GT 생성 함수
	def generate_heatmap_gt(self, joint_location, image_size=[64,64],sigma=2):
		heatmap_gt = np.zeros((len(joint_location),image_size[1], image_size[0]), dtype=np.float32)
		for i, joint in enumerate(joint_location):
			heatmap_gt[i, :, :] = self.generate_gaussian_heatmap(joint, image_size, sigma)
		return heatmap_gt
	
	#TODO : gaussian code 
	def get_gaussian_heatmap(self,json_path):

		"""
		# 관절 위치, 가우시안 분포 크기, 이미지 크기를 지정합니다.
		joint_location = [(x, y) for x, y in label_data['keypoints']]
		sigma = 1
		image_size = (512, 512)

		# heatmap GT를 생성합니다.
		heatmap_gt = generate_heatmap_gt(image_size, joint_location, sigma)
		"""
		res=None
		fisheye_joint_labels = self.get_fisheye_2d_joints(json_path,resize=(64,64))
		res = self.generate_heatmap_gt(fisheye_joint_labels)
		return res

	def get_info(self,json_path):
		temp_json=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())
		pass

		
	def get_shape(self):
		pass

	def __len__(self):
		return self.IMAGE['total_len']

	def __getitem__(self, index):
		# assert self.transform is not None
		img_path=get_path(index,self.IMAGE)
		json_path=get_path(index,self.JSON)
		seg_path=get_path(index,self.SEGMAP)
		depth_path=get_path(index,self.DEPTH)
	   
		image=self.get_rgba(img_path)
		joints_3d_world = self.get_3d_joints(json_path,cam_coord=False)
		joints_3d_cam = self.get_3d_joints(json_path,cam_coord=True)
		# joints_2d=self.get_2d_joints(json_path)
		seg_image=self.get_rgba(seg_path)
		camera_info=self.get_camera(json_path)
		depth_map=self.get_depth(depth_path)
		fisheye_joints_2d=self.get_fisheye_2d_joints(json_path,resize=(512,512))
		heatmap=self.get_gaussian_heatmap(json_path)

		res={}
		if self.is_train:
			res.update({
				'info' : img_path,
				'image': image,
				'joints_3d' : joints_3d_world,
				'joints_3d_cam' : joints_3d_cam,
				# 'joints_2d' : joints_2d, 
				'seg_image': seg_image,
				'depth' : depth_map,
				'camera_info': camera_info,
				'fisheye_joints_2d' : fisheye_joints_2d,
				'heatmap' : heatmap
			})
			return res
		res.update({
			'image':image
		})
		return res

	def split_dataset(self) -> Tuple[Subset, Subset]:
		"""
		데이터셋을 train 과 val 로 나눕니다,
		pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
		torch.utils.data.Subset 클래스 둘로 나눕니다.
		구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
		"""
		n_val = int(len(self) * self.val_ratio)
		n_train = len(self) - n_val
		train_set, val_set = random_split(self, [n_train, n_val])
		return train_set, val_set