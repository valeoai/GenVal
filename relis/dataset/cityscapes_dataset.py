'''
https://github.com/naver/oasis/blob/master/dataset/cityscapes_dataset.py
'''
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import glob

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
import cv2

import sys
sys.path.append("..")
from transformation_ops import TransfOps
from path_dicts import *


class Cityscapes(data.Dataset):
    def __init__(self, root, city_list, cond_list, set='val',
                num_epochs=1, do_augm=False, num_transf=3,
                transf_list=['identity', 'RGB_rand', 'brightness', 'color', 'contrast', 'black_and_white'],
                return_label_name=False):
        """
            params

                root : str
                    Path to the data folder'

        """

        self.class_conversion_dict = {
                7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
                23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

        self.root = root
        self.augmented_root = CITYSCAPES_AUG_ROOT
        self.city_list = city_list
        self.cond_list = cond_list
        
        self.return_label_name = return_label_name

        self.files = []
        self.label_files = []

        if do_augm:
            self.augment = True
            self.transf_ops = TransfOps(transformation_list=transf_list)
            self.num_transf = num_transf
        else:
            self.augment = False

        for cond in cond_list:
            if cond not in ['clean', 'augmented', 'fog', 'night', 'rain', 'snow', 'india']:
                raise ValueError('Unknown conditions [supported are clean, augmented, fog, night, rain, snow and india]')

        assert len(cond_list) == len(city_list)

        self.num_imgs_per_seq = []

        self.images_root = self.root

        for city, cond in zip(self.city_list, self.cond_list):

            if city in ['berlin', 'bielefeld', 'bonn',
                    'leverkusen', 'mainz', 'munich']:
                self.set = 'test'

            elif city in ['frankfurt', 'lindau', 'munster']:
                self.set = 'val'

            else:
                self.set = 'train'


            if cond != 'augmented':
                list_of_images_file = glob.glob(f'{self.images_root}/leftImg8bit/{self.set}/{city}/*.png')
                self.img_names = [x.split(f'/{self.set}/')[-1] for x in list_of_images_file]
                list_of_label_images_file =  glob.glob(f'{self.images_root}/gtFine/{self.set}/{city}/*labelIds.png')
                self.label_img_names = [x.split(f'/{self.set}/')[-1] for x in list_of_label_images_file]

#             self.img_names = [i_id.strip() for i_id in open(list_of_images_file)]
            if cond == 'clean':
                pass

            elif cond == 'augmented':
                self.img_names = glob.glob(f'{self.augmented_root}/{city}/images/*.png')
                self.label_img_names = glob.glob(f'{self.augmented_root}/{city}/labels/*.png')
                

            self.num_imgs_per_seq.append(len(self.img_names))

            for name in sorted(self.img_names):
                if cond == 'clean':
                    img_path = os.path.join(
                            self.images_root,
                            f'leftImg8bit/{self.set}/{name}')

                elif cond == 'augmented':
                    img_path, name = name, name.split('/')[-1]


                self.files.append({
                    'img': img_path, # used path
                    'name': name # just the end of the path
                })

            for name in sorted(self.label_img_names):
                if cond == 'augmented':
                    img_path, name = name, name.split('/')[-1]
                else:
                    img_path = os.path.join(self.root, f'gtFine/{self.set}/{name}')
                self.label_files.append({
                    'label_img': img_path, # used path
                    'label_name': name # just the end of the path
                })

        self.annotation_path_list = [_['label_img'] for _ in self.label_files]


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        image = np.asarray(image)

        # ---- code for augmentation ---------------		
        if self.augment:
            image = np.expand_dims(image, 0)
            image, _, _ = self.transf_ops.transform_dataset(
                    image, transf_string='random_'+str(self.num_transf))
            image = np.squeeze(image).astype(np.float32) 
		# ------------------

        name = self.files[index]['img']

        label = Image.open(self.label_files[index]['label_img'])#.convert('RGB')
        label = np.asarray(label)
        label_name = self.label_files[index]['label_name']

        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            label_copy[label == k] = v

        size = image.shape

        if self.return_label_name:
            label_path = self.label_files[index]['label_img']
            return image.copy(), label_copy.copy(), np.array(size), name, label_path
        
        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    pass
