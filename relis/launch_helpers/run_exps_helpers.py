"""
Code to generate sequences used to replicate the experiments
presented in the main paper. All functions take in input lists
for datasets/conditions/scenes and append key words for specific
experiments to them.

Adapted from https://github.com/naver/oasis/blob/master/run_exps_helpers.py
"""

import glob
import numpy as np
import numpy.random as npr
import os
import sys

sys.path.append("..")
from path_dicts import *

def update_cityscapes_lists(mode, trg_dataset_list, scene_list, cond_list):

    train_cities = ['aachen', 'dusseldorf', 'krefeld', 'ulm',
                'bochum', 'erfurt', 'monchengladbach', 'weimar',
                'bremen', 'hamburg', 'strasbourg', 'zurich',
                'cologne', 'hanover', 'stuttgart',
                'darmstadt', 'jena', 'tubingen']

    # Changing val and test splits for CS since test labels are all non-valid...
    val_cities = ['munster', 'lindau']
    test_cities = ['frankfurt']

    if mode == 'train-1-clean':
        scene_list_ = train_cities
        cond_list_ = ['clean'] * len(scene_list_)

    elif mode == 'val-1-clean':
        scene_list_ = val_cities
        cond_list_ = ['clean'] * len(scene_list_)
        
    elif mode == 'test-1-clean':
        scene_list_ = test_cities
        cond_list_ = ['clean'] * len(scene_list_)
        
    elif mode == 'val-1-augmented':
        scene_list_ = val_cities
        cond_list_ = ['augmented'] * len(scene_list_)
        
    else:
        raise ValueError('Unknown mode')

    trg_dataset_list_ = ['Cityscapes'] * len(scene_list_)

    trg_dataset_list += trg_dataset_list_
    scene_list += scene_list_
    cond_list += cond_list_

    return None


def update_acdc_lists(mode, trg_dataset_list, scene_list, cond_list):

    dset_root = os.path.join(ACDC_ROOT, 'gt_trainval/gt/')
    
    if mode == 'val':
        path_list = glob.glob(dset_root + '/*/val/*')
        scene_list_ = [x.split('/')[-1] for x in path_list]
        cond_list_ = [x.split('/')[-3] for x in path_list]
        
    elif mode == 'train':
        path_list = glob.glob(dset_root + '/*/train/*')
        scene_list_ = [x.split('/')[-1] for x in path_list]
        cond_list_ = [x.split('/')[-3] for x in path_list]
    
    else:
        raise ValueError('Unknown mode')

    trg_dataset_list_ = ['ACDC'] * len(scene_list_)

    trg_dataset_list += trg_dataset_list_
    scene_list += scene_list_
    cond_list += cond_list_

    return None

def update_idd_lists(mode, trg_dataset_list, scene_list, cond_list):

    if mode == 'train_clean':
#         scene_list_ = os.listdir(os.path.join(dset_root, 'gtFine/train/'))

        # We randomly selected a subset of training scenes so evaluation is not so heavy
        scene_list_ = ['265', '431', '175', '528', '39', '137', '538', '499', '7', '376',
                       '37', '209', '306', '413', '380', '374', '472', '523', '446',
                       '121', '171', '488', '278', '411', '312', '453', '110', '285',
                       '46', '158', '426', '544', '230', '41', '520', '322', '95']

    elif mode == 'val_clean':
        scene_list_ = os.listdir(os.path.join(IDD_ROOT, 'gtFine/val/'))
    
    # # For quick tests 
    # if mode == 'train_clean':
    #     scene_list_ = ['265']
    # elif mode == 'val_clean':
    #     scene_list_ = ['267']

    trg_dataset_list_ = ['IDD'] * len(scene_list_)
    cond_list_ = ['clean'] * len(scene_list_)
    
    trg_dataset_list += trg_dataset_list_
    scene_list += scene_list_
    cond_list += cond_list_

    return None


