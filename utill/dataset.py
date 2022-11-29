import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json

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
    def __init__(self,opt,mode = "train") -> None:
        self.opt=opt
        self.mode = mode
        self.root = self.opt.dataroot
        self.load_size = self.opt.loadsize    
        self.is_train = (mode=='train')
        
        self.IMAGE = load_data(self.root,'rgba')
        self.subject_names=self.IMAGE.keys()

        if not self.mode:
            self.JSON = load_data(self.root,'json')
            self.SEGMAP = load_data(self.root,'objectId')
            self.SMPL = os.path.join(self.root,'smpl')

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # # augmentation
        # self.aug_trans = transforms.Compose([
        #     transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
        #                            hue=opt.aug_hue)
        # ])
        


        """
        print('load images ...')    
        for img_info,ann_list in tqdm(zip(self.img_infos,self.anns),total=len(self.img_infos)):
            img_name = img_info[0]['file_name']
            # get PIL image
            image = Image.open(os.path.join(image_root,img_name))
            image = ImageOps.exif_transpose(image).convert('L')
            
            # ann --> GT_mask
            y = np.zeros((image.size[1],image.size[0]))
            for ann in ann_list:
                y[coco.annToMask(ann[0]) == 1] = ann[0]['category_id']

            # image preprocess (PIL --> numpy)
            image, y = custom_utils.img_rotate(image,y)

            # get ocr out
            ocr_out = get_ocr(Image.fromarray(image),api_url)

            ###### ocr_list to mask ---> concat to image c4:all text, c5: wifi key, c6: id key, c7: pw key
            ocr_coco = ocr_to_coco(ocr_out,os.path.join(self.image_root,img_name),(image.shape[0],image.shape[1]))
            c2 = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=False)
            c3 = coco_to_mask(ocr_coco,image.shape,key_list=key_list,get_each_mask=False)

            _,mask_out = coco_to_mask(ocr_coco,image.shape,key_list=None,get_each_mask=True)

            t = torchvision.transforms.ToPILImage()
            c2 = np.array(t(c2))
            c3 = np.array(t(c3))
            
            self.x_list.append(image)
            self.ocr_lists.append((mask_out,ocr_out))
            self.y_list.append(y)
            self.c_list.append((c2,c3))
        """

    def get_rgba(self,img_path):
        image_=Image.open(img_path).convert('RGB')
        return self.to_tensor(image_)
        
    def get_camera(self,json_path):
        pass

    def get_joints(self,json_path):
        temp_json=None
        with open(json_path,'r') as f:
            temp_json=json.loads(f.read())
        
        return temp_json['pts3d_fisheye']

    def get_shape(self):
        pass

    def __len__(self):
        return self.IMAGE['total_len']

    def __getitem__(self, index):
        img_path=get_path(index,self.IMAGE)
        json_path=get_path(index,self.JSON)
        seg_path=get_path(index,self.SEGMAP)
        res={}
        if self.is_train:
            res.update({
                'image': self.get_rgba(img_path),
                'joints': self.get_joint(json_path),
                'seg_image':self.get_rgba(seg_path),
                'camera_info':self.get_camera(json_path)
            })
            return res
        res.update({
            'image':self.get_rgba(img_path)
        })
        return res
