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
    for folder in glob.glob(os.path.join(dataroot,'*')):
        subject_name=folder.split('\\')[-1]
        path_dict={}
        for env in glob.glob(os.path.join(folder,'*')):
            env_name=env.split('\\')[-1]
            path_=glob.glob(os.path.join(env,'cam_down',data_type,extension))
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
		
		if self.is_train:
			self.JSON = load_data(self.root,'json')
			self.SEGMAP = load_data(self.root,'objectId')
			self.SMPL = os.path.join(self.root,'smpl')

		# PIL to tensor
		self.transform = None

		# # augmentation
		# self.aug_trans = transforms.Compose([
		#     transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
		#                            hue=opt.aug_hue)
		# ])

	def set_transform(self, transform):
		self.transform = transform  
			
	def get_rgba(self,img_path):
		image_=Image.open(img_path).convert('RGB')
		return self.transform(image_)
		
	def get_camera(self,json_path):
		temp_json=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())
		
		return temp_json['camera']

	def get_joints(self,json_path):
		temp_json=None
		with open(json_path,'r') as f:
			temp_json=json.loads(f.read())
		x=torch.tensor(temp_json['pts3d_fisheye'][0])
		y=torch.tensor(temp_json['pts3d_fisheye'][1])
		z=torch.tensor(temp_json['pts3d_fisheye'][2])
		pelvis_x=torch.tensor(x[2])
		pelvis_y=torch.tensor(y[2])
		pelvis_z=torch.tensor(z[2])
		x=(x-pelvis_x)*.01
		y=(y-pelvis_y)*.01
		z=(z-pelvis_z)*.01
		res=[x,y,z]
		res = torch.stack(res,0)
		return res

	def get_shape(self):
		pass

	def __len__(self):
		return self.IMAGE['total_len']

	def __getitem__(self, index):
		assert self.transform is not None
		img_path=get_path(index,self.IMAGE)
		json_path=get_path(index,self.JSON)
		seg_path=get_path(index,self.SEGMAP)
		
		image=self.get_rgba(img_path)
		joints=self.get_joints(json_path)
		seg_image=self.get_rgba(seg_path)
		camera_info=self.get_camera(json_path)

		res={}
		if self.is_train:
			res.update({
				'image': image,
				'joints': joints,
				'seg_image': seg_image,
				'camera_info': camera_info
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