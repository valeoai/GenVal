'''
Adapted from https://github.com/naver/oasis/blob/master/main_adapt.py
'''

import sys
import os
import glob
import matplotlib.pyplot as plt
import random
import json
import copy
import argparse
import copy
import pickle
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD

from image_helpers import ImageOps

from path_dicts import *


class SolverOps:

    def __init__(self, args):

        self.args = args


    def create_augmented_dataset(self):

        if self.args.batch_size != 1:
            raise NotImplementedError("Code only supported for BS = 1 for the moment")
        for epoch in range(self.args.epochs):
            for i_iter, trg_batch in enumerate(self.trg_eval_loader):

                # Collect one batch (single image if bs=1)
                trg_image, _, _, trg_image_name, label_name = trg_batch
                trg_image = np.array(trg_image.squeeze()) # Convert image from torch to np array.
                trg_image_name = trg_image_name[0]
                label_name = label_name[0]
                label = Image.open(label_name)
                

                ########### SAVING IMAGES ########################

                name = trg_image_name.split('/')[-1].split('.')[0]
                augmented_image_name = os.path.join(self.args.augmented_dataset_dir, 'images',
                                                    f'{name}_aug{epoch}.png')
                Image.fromarray(trg_image.astype('uint8')).save(augmented_image_name)
                
                augmented_label_name = os.path.join(self.args.augmented_dataset_dir, 'labels',
                                                    f'{name}_aug{epoch}.png')
                label.save(augmented_label_name)


        print('Done creating augmented dataset.')

        with open(os.path.join(self.args.augmented_dataset_dir, self.DONE_name),'wb') as f:
            print('Saving end of training file')

   
    def setup_dataset_folder(self):


        self.DONE_name = f'experiment.DONE'

        # check if experiment/testing was done already -------------------------------
        if os.path.isfile(
                    os.path.join(self.args.augmented_dataset_dir, self.DONE_name)) \
                and not self.args.force_redo:
            print('DONE file present -- evaluation has already been carried out')
            print(os.path.join(self.args.augmented_dataset_dir, self.DONE_name))
            exit(0)
        # -----------------------------------------------------------------------------

        print(f'EXP ---> {self.args.augmented_dataset_dir}')

        labels_dir = os.path.join(self.args.augmented_dataset_dir, 'labels')
        images_dir = os.path.join(self.args.augmented_dataset_dir, 'images')
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)


    def setup_target_data_loader(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        scene_list = [self.args.scene]
        cond_list = [self.args.cond]
            

        if self.args.trg_dataset=='Cityscapes':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_ROOT,
                    scene_list, cond_list, do_augm=True,
                    num_transf=self.args.num_transf,
                    return_label_name=True)

        else:
            raise ValueError(f'Unknown dataset {self.args.dataset}')

        self.trg_eval_loader = data.DataLoader(
                self.trg_parent_set, batch_size=self.args.batch_size,
                shuffle=False, pin_memory=True)


if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser()

    # main experiment parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to {Cityscapes}")
    parser.add_argument("--scene", type=str, default='munchen',
                        help="Scene, depends on specific datasets.")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")

    # for augmentations
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of 'epochs' i.e. different augmentations per image.")
    parser.add_argument("--num_transf", type=int, default=3,
                        help="Number of transformations to concatenate during augmentation.")
    parser.add_argument("--augmented_dataset_dir", type=str, default='results/debug_dataset/',
                        help="Folder where to store the augmented dataset.")
    
    
    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)

    npr.seed(args.seed)

    solver_ops = SolverOps(args)

    print('Setting up dataset folder')
    solver_ops.setup_dataset_folder()

    print('Setting up data loader')
    solver_ops.setup_target_data_loader()

    print('Generating augmented dataset')
    solver_ops.create_augmented_dataset()

