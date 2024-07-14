import os
import glob
import pickle
import json
import copy

import torch
from torch.utils import data
from torchvision.transforms import ToTensor

import numpy as np
import numpy.random as npr

from sklearn.utils import shuffle

from PIL import Image

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from launch_helpers import run_exps_helpers
from path_dicts import *

from .cityscapes_dataset import Cityscapes
from .acdc_dataset import ACDC
from .idd_dataset import IDD



class MixedDatasets:
    def __init__(self, scene_list, cond_list, dset_list):
        self.scene_list = scene_list
        self.cond_list = cond_list
        self.dset_list = dset_list
        self.unique_dsets = list(set(dset_list))
        
        self.setup_datasets()
        
        self.retrieve_labels_images_paths()
        self.to_tensor = ToTensor()
        
        
    def setup_datasets(self):

        """
        Function to create pytorch dataloaders for the
        target domains selected by the user
        """

        scene_list = self.scene_list
        cond_list = self.cond_list
        dset_list = self.dset_list
        unique_dsets = self.unique_dsets


        self.trg_parent_sets_dict = {}
        for dset in unique_dsets:
            indx = [i for i, value in enumerate(dset_list) if dset == value]
            dset_conds = [cond_list[i] for i in indx]
            dset_scenes = [scene_list[i] for i in indx]

            if dset=='Cityscapes':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_ROOT,
                        dset_scenes, dset_conds)

            elif dset=='ACDC':
                trg_parent_set = ACDC(
                        ACDC_ROOT, dset_scenes, dset_conds,
                        batch_size=1)

            elif dset=='IDD':
                trg_parent_set = IDD(
                        IDD_ROOT, dset_scenes, batch_size=1)

            else:
                raise ValueError(f'Unknown dataset {dset}')

            self.trg_parent_sets_dict[dset] = trg_parent_set
            
            
            
    def retrieve_labels_images_paths(self):
        """
        Function to retrieve the paths to labels and images
        based on the condition and scene list.
        """

        scene_list = self.scene_list
        cond_list = self.cond_list
        dset_list = self.dset_list

        # Sorted (paired) preds and labels lists
        labels_list = []
        images_list = []
        all_dsets_list = []
        
        for scene, cond, dset in zip(scene_list, cond_list, dset_list):
            if cond == 'clean':
                cond = ''
            scene = f'/{scene}/'
            # Unsorted all labels list
            all_labels = self.trg_parent_sets_dict[dset].annotation_path_list
            all_images = self.trg_parent_sets_dict[dset].files
            all_images = [x['img'] for x in all_images]

            new_images = sorted([x for x in all_images if (scene in x and cond in x)])
            new_labels = sorted([x for x in all_labels if (scene in x and cond in x)])
            labels_list += new_labels
            images_list += new_images
            all_dsets_list += [dset]*len(new_images)

        self.images_list = images_list
        self.labels_list = labels_list
        self.all_dsets_list = all_dsets_list
        assert len(images_list) == len(labels_list)
        
    def random_split_train_test(self, percentage_val=0.2, total_images=500):
        
        images_list, labels_list, all_dsets_list = shuffle(self.images_list, self.labels_list, self.all_dsets_list)
        
        if len(images_list) > total_images:
            images_list = images_list[:total_images]
            labels_list = labels_list[:total_images]
            all_dsets_list = all_dsets_list[:total_images]
            
        
        val_set = copy.deepcopy(self)
        train_set = copy.deepcopy(self)
        
        val_idx = round(percentage_val * len(images_list))
        
        val_set.images_list = images_list[:val_idx]
        val_set.labels_list = labels_list[:val_idx]
        val_set.all_dsets_list = all_dsets_list[:val_idx]
        
        train_set.images_list = images_list[val_idx:]
        train_set.labels_list = labels_list[val_idx:]
        train_set.all_dsets_list = all_dsets_list[val_idx:]
        
        return train_set, val_set
    
        
    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, index):
        image = Image.open(self.images_list[index])
        image = self.to_tensor(image)
        
        image_path = self.images_list[index]
        
        label = Image.open(self.labels_list[index])
        label = np.asarray(label)
        
        dset = self.all_dsets_list[index]
        
        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        
        for k, v in self.trg_parent_sets_dict[dset].class_conversion_dict.items():
            label_copy[label == k] = v
        
        return image, label_copy, image_path 
        