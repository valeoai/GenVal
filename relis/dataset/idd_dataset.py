'''
Adapted from https://github.com/naver/oasis/blob/master/dataset/cityscapes_dataset.py
'''
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import glob
import pickle

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
import cv2


'''
Conversion Cityscapes to IDD:
    
    Cityscapes classes:
        0 road
        1 sidewalk
        2 building
        3 wall
        4 fence
        5 pole
        6 light
        7 sign
        8 vegetation
        9 terrain # NOT IN IDD!!!
        10 sky
        11 person
        12 rider
        13 car
        14 truck
        15 bus
        16 train
        17 motocycle
        18 bicycle
        
    IDD classes:
        0 road
        1 parking
        1 drivable fallback
        2 sidewalk
        3 rail track
        3 non-drivable fallback
        4 person
        4 animal
        5 rider
        6 motorcycle
        7 bicycle
        8 autorickshaw
        9 car
        10 truck
        11 bus
        12 caravan
        12 trailer
        12 train
        12 vehicle fallback
        13 curb
        14 wall
        15 fence
        16 guard rail
        17 billboard
        18 traffic sign
        19 traffic light
        20 pole
        20 polegroup
        21 obs-str-bar-fallback
        22 building
        23 bridge
        23 tunnel
        24 vegetation
        25 sky
        25 fallback background
        255 unlabeled
        255 ego vehicle
        255 rectification border
        255 out of roi
        255 license plate
        

'''

class IDD(data.Dataset):
    def __init__(self, root, scene_list, batch_size = 1):
        """
            params

                root : str
                    Path to the data folder.

        """

        # TO DO
        self.class_conversion_dict = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                                     5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}

        self.root = root
        self.scene_list = scene_list

        self.batch_size = batch_size

        self.files = []
        self.label_files = []

        self.num_imgs_per_seq = []

        for scene in self.scene_list:

            self.img_paths = glob.glob(os.path.join(self.root, 'leftImg8bit/*/', scene, '*leftImg8bit.*'))
            self.img_paths = sorted(self.img_paths)

            self.label_img_paths = glob.glob(os.path.join(self.root, 'gtFine/*/', scene, '*labellevel3Ids*.png'))
            self.label_img_paths = sorted(self.label_img_paths)

            print(f'Retrieving {scene} ({len(self.img_paths)} images)')

            self.num_imgs_per_seq.append(len(self.img_paths))

            for img_path in self.img_paths:
                name = img_path.split('/')[-1]
                self.files.append({
                    'img': img_path, # used path
                    'name': name # just the end of the path
                })

            for label_img_path in self.label_img_paths:
                name = label_img_path.split('/')[-1]
                self.label_files.append({
                    'label_img': label_img_path, # used path
                    'label_name': name # just the end of the path
                })

        self.annotation_path_list = [_['label_img'] for _ in self.label_files]
        self.img_path_list = [_['img'] for _ in self.files]

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        image = np.asarray(image, np.float32)
        name = self.files[index]['img']

        label = Image.open(self.label_files[index]['label_img'])#.convert('RGB')
        label = np.asarray(label)
        label_name = self.label_files[index]['label_name']

        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            label_copy[label == k] = v

        size = image.shape

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    pass

