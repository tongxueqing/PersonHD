from __future__ import division
import torch
import torchvision.transforms as transforms
from .base_dataset import *
import cv2
import numpy as np
import os
import util.io as io
from albumentations  import ColorJitter,ChannelShuffle
import random

aug_color_jitter_face=ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False,p=0.5)

aug_color_jitter=ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, always_apply=False,p=0.5)

aug_channel_shuffle=ChannelShuffle(p=0.5)
class PoseTransferParsingDataset(BaseDataset):
    def name(self):
        return 'PoseTransferParsingDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.data_root = opt.data_root
        self.split = split
        #############################
        # set path / load label
        #############################
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))

        self.seg_cihp_dir = os.path.join(opt.data_root, opt.seg_dir)
        

        #############################
        # create index list
        #############################
        # self.id_list = data_split[split][:50000] if split in data_split.keys() else data_split['test'][:2000]
        self.id_list = data_split[split] if split in data_split.keys() else data_split['test'][:2000]
        self._len = len(self.id_list)
        #############################
        # other
        #############################
        # here set debug option
        if opt.debug:
            self.id_list = self.id_list[0:32]
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.to_pil_image = transforms.ToPILImage()
        self.pil_to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.2)

    def set_len(self, n):
        self._len = n

    def __len__(self):
        if hasattr(self, '_len') and self._len > 0:
            return self._len
        else:
            return len(self.id_list)

    def to_tensor(self, np_data):
        return torch.Tensor(np_data.transpose((2, 0, 1)))

    def read_image(self, sid):
        fn = os.path.join(self.img_dir, sid + '.jpg')
        # print(fn)
        # print(os.path.exists(fn))
        img = cv2.imread(fn).astype(np.float32) 
        # img = cv2.resize(img,(256,256))
        img = img/ 255.
        img = img[..., [2, 1, 0]]
        return img
    
    

    def read_image_jitter(self, sid,seg,seed):
    '''
    This is the implement of Semantic Enhanced Part-wise Augmentation. As described in Section4.1.  
    '''
        fn = os.path.join(self.img_dir, sid + '.jpg')
        img = cv2.imread(fn).astype(np.float32) 
        
        img = img/ 255.
        random.seed(seed)

        #13 face: for face we just do litter jitter
        #0 background: no change to background
        img_jitter_face=aug_color_jitter_face(image=img)['image']
        img_result=img*(seg==0)+img_jitter_face*(seg==13)
        for i in range(1,13):
            img_jitter=aug_color_jitter(image=img)['image']
            img_shuffle=aug_channel_shuffle(image=img_jitter)['image']
            img_result+=img_shuffle*(seg==i)
        
        # make sure left-right consistency
        #14 /15 left arm / right arm
        #16/ 17 left leg / right leg
        #18/ 19 left shoe / right shoe
        img_jitter=aug_color_jitter(image=img)['image']
        img_shuffle=aug_channel_shuffle(image=img_jitter)['image']
        img_result+=img_shuffle*((seg==14)|(seg==15))

        img_jitter=aug_color_jitter(image=img)['image']
        img_shuffle=aug_channel_shuffle(image=img_jitter)['image']
        img_result+=img_shuffle*((seg==16)|(seg==17))

        img_jitter=aug_color_jitter(image=img)['image']
        img_shuffle=aug_channel_shuffle(image=img_jitter)['image']
        img_result+=img_shuffle*((seg==18)|(seg==19))

        img = img[..., [2, 1, 0]]
        
        return img



    def read_seg_pred_cihp(self, sid1, sid2):
        fn = os.path.join(self.seg_cihp_pred_dir, sid1 + '___' + sid2 + '.png')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        seg = cv2.resize(seg,(256,256))
        seg = seg[..., np.newaxis]
        return seg

    def read_seg_cihp(self, sid):
        fn = os.path.join(self.seg_cihp_dir, sid + '.png')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # seg = cv2.resize(seg,(256,256))
        seg = seg[..., np.newaxis]
        return seg
        
    def resize_(self, x):
        y=x*[256/190,256/110]
        y=y.astype(np.int)
        return y        

    def __getitem__(self, index):
        sid1, sid2 = self.id_list[index]
        ######################
        # load data
        ######################
        # img_1 = self.read_image(sid1)
        # img_2 = self.read_image(sid2)
        
        seg_cihp_label_1 = self.read_seg_cihp(sid1)
        # seg_cihp_label_2 = self.read_seg_cihp(sid2) if self.split=='train' else self.read_seg_pred_cihp(sid1, sid2)
        seg_cihp_label_2 = self.read_seg_cihp(sid2)

        ## set the seed to make sure the image pair change consistently
        seed=np.random.randint(0,100)
        img_1 = self.read_image_jitter(sid1,seg_cihp_label_1,seed) if self.split=='train' else self.read_image(sid1)

        img_2 = self.read_image_jitter(sid2,seg_cihp_label_2,seed) if self.split=='train' else self.read_image(sid2)

        joint_c_1 = np.array(self.pose_label[sid1])
        
        joint_c_2 = np.array(self.pose_label[sid2])
        
        

        h, w = self.opt.image_size
        ######################
        # pack output data
        ######################
        joint_1 = kp_to_map(img_sz=(w, h), kps=joint_c_1, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        joint_2 = kp_to_map(img_sz=(w, h), kps=joint_c_2, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        seg_cihp_1 = seg_label_to_map(seg_cihp_label_1, nc=20)
        seg_cihp_2 = seg_label_to_map(seg_cihp_label_2, nc=20)

        data = {
            'img_1': self.tensor_normalize_std(self.to_tensor(img_1)),
            'img_2': self.tensor_normalize_std(self.to_tensor(img_2)),
            'joint_1': self.to_tensor(joint_1),
            'joint_2': self.to_tensor(joint_2),
            'seg_cihp_1': self.to_tensor(seg_cihp_1),
            'seg_cihp_2': self.to_tensor(seg_cihp_2),
            'id_1': sid1,
            'id_2': sid2
        }
        return data