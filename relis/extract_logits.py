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
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

from PIL import Image

# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD

from image_helpers import ImageOps

from path_dicts import *


class SolverOps:

    def __init__(self, args):

        self.args = args

        # this is taken from the AdaptSegnet repo
        with open('./dataset/cityscapes_list/info.json', 'r') as f:
            cityscapes_info = json.load(f)

        self.args.num_classes = 19
        self.args.name_classes = cityscapes_info['label']

        print(f'Number of classes: {self.args.num_classes}')

        assert len(self.args.cond.split('-')) == len(self.args.scene.split('-'))

        self.image_ops = ImageOps()

        w_trg, h_trg = map(int, self.args.input_size.split(','))
        self.input_size = (w_trg, h_trg)


    def extract_logits(self):

        """
        Method to extract features from a model sample by sample on a given
        sequence and save them.
        All parameters setup by the user (args).
        """

        cudnn.enabled = True
        gpu = self.args.gpu
        #torch.use_deterministic_algorithms(True) # TODO fix weird error
        cudnn.benchmark = True


        self.args.num_steps = len(self.trg_eval_loader)
        
        if self.args.samples_per_image == -1:
            self.args.samples_per_image = self.input_size[0] * self.input_size[1]

        if self.args.batch_size != 1:
            raise NotImplementedError("Code only supported for BS = 1 for the moment")        
        for i_iter, trg_batch in enumerate(self.trg_eval_loader):

            # Collect one batch (single image if bs=1)
            trg_image, trg_labels, _, trg_image_name = trg_batch
            trg_image = np.array(trg_image.squeeze()) # Convert image from torch to np array for mmseg.
            trg_image_name = trg_image_name[0]
            
            if 'Mask' in self.args.model_arch: # Use Detectron2 library
                from detectron2.data.detection_utils import read_image
                image = read_image(trg_image_name, format="BGR")
                
                logits_cpu = self.model.get_mask_pred_logits(image) # Directly extracts logits on cpu
                pred_masks = logits_cpu['pred_masks'][0]
                num_queries = pred_masks.shape[0]
                pred_masks = pred_masks.permute(1,2,0).view(-1, num_queries).numpy()
                samples_indices = np.random.choice(range(pred_masks.shape[0]), self.args.samples_per_image, replace=False)
                logits_cpu['pred_masks'] = pred_masks[samples_indices]
                logits_cpu['pred_logits'] = logits_cpu['pred_logits'][0]
                
                
            else: # Use mmsegmentation model
                from mmsegmentation.mmseg.apis import inference_segmentor
                logits = inference_segmentor(self.model, trg_image_name, output_logits=True, pre_softmax=True)
                logits_cpu = logits.cpu().data[0] #1 torch.Size([C, H, W])
                # process the features for saving
                logits_cpu = logits_cpu.permute(1,2,0).view(-1, self.args.num_classes).numpy() #2 (H * W, C)
                samples_indices = np.random.choice(range(logits_cpu.shape[0]), self.args.samples_per_image, replace=False)
                logits_cpu = logits_cpu[samples_indices]

            ########### SAVING FEATURES ########################
            
            image_name = trg_image_name.split('/')[-1].split('.')[0]
            
            torch.save(logits_cpu, os.path.join(self.output_dir, f'{image_name}_logits.pt'))
            torch.save(samples_indices, os.path.join(self.output_dir, f'{image_name}_indices.pt'))
            
            ##################################################

        print('End of evaluation.')

        with open(os.path.join(self.output_dir, self.DONE_name),'wb') as f:
            print('Saving end of training file')

            
    def build_model(self):
        # Create network
        config = mmseg_models_configs[self.args.model_arch]
        checkpoint = mmseg_models_checkpoints[self.args.model_arch]
        if 'Mask' in self.args.model_arch: # Use Detectron2 library
            
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config

            cfg = get_cfg()
            add_deeplab_config(cfg)
            if 'Mask2' in self.args.model_arch:
                from Mask2Former.mask2former import add_maskformer2_config
                add_maskformer2_config(cfg)
            else:
                from MaskFormer.mask_former import add_mask_former_config
                add_mask_former_config(cfg)
            cfg.merge_from_file(config)
            cfg.freeze()
            self.model = DefaultPredictor(cfg) # To be used on a single GPU
            
        else: # Use MMSegmentation library
            
            from mmsegmentation.mmseg.apis import init_segmentor
            # Create network
            config = mmseg_models_configs[self.args.model_arch]
            checkpoint = mmseg_models_checkpoints[self.args.model_arch]
            self.model = init_segmentor(config, checkpoint,
                                        device=f'cuda:{self.args.gpu}')
            # Set model decoder to provide features
            self.model.decode_head.provide_features = True

            # Set up config of the model to process the dataset
            self.model.cfg.test_pipeline = [
                                            {'type': 'LoadImageFromFile'},
                                            {'type': 'MultiScaleFlipAug',
                                                'img_scale': (self.input_size[0], self.input_size[1]),
                                                'flip': False,
                                                'transforms': [
                                                    {'type': 'Resize', 'keep_ratio': True},
                                                    {'type': 'RandomFlip'},
                                                    {'type': 'Normalize',
                                                        'mean': [123.675, 116.28, 103.53], # TODO: Should we adapt it to target dsets?
                                                        'std': [58.395, 57.12, 57.375],
                                                        'to_rgb': True},
                                                    {'type': 'ImageToTensor', 'keys': ['img']},
                                                    {'type': 'Collect', 'keys': ['img']}
                                                ]
                                            }
                                        ]
        print('Done')
        

    def setup_experiment_folder(self):

        """
        Method to define model folder's name and create it, and to
        define the name of the output files created at end of training.
        """

        trg_sub_folder = f'{self.args.trg_dataset}_{self.args.scene}_{self.args.cond}'
        method_sub_folder = f'extracted_logits'
        model_arch_sub_folder = self.args.model_arch
        num_samples = f'num_samples_{self.args.samples_per_image}'

        self.DONE_name = f'experiment.DONE'
        
        self.output_dir = os.path.join(
                self.args.root_exp_dir, self.args.src_dataset,
                model_arch_sub_folder, trg_sub_folder, method_sub_folder, num_samples)

        # check if experiment/testing was done already -------------------------------
        if os.path.isfile(
                os.path.join(self.output_dir, self.DONE_name)) \
                    and not self.args.force_redo:
            print('DONE file present -- evaluation has already been carried out')
            print(os.path.join(self.output_dir, self.DONE_name))
            exit(0)
        # -----------------------------------------------------------------------------


        print(f'EXP ---> {self.output_dir}')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def setup_target_data_loader(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        # (can also be a single environment)
        scene_list = [self.args.scene]
        cond_list = [self.args.cond]
            

        if self.args.trg_dataset=='Cityscapes':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_ROOT,
                    scene_list, cond_list)
            
        elif self.args.trg_dataset=='Cityscapes_fog':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_FOG,
                    scene_list, cond_list)
        
        elif self.args.trg_dataset=='Cityscapes_night':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_NIGHT,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_rain':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_RAIN,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_snow':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_SNOW,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_india':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_INDIA,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_all':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_ALL,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='ACDC':
            self.trg_parent_set = ACDC(
                    ACDC_ROOT, scene_list, cond_list,
                    batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='IDD':
            self.trg_parent_set = IDD(
                    IDD_ROOT, scene_list, batch_size=self.args.batch_size)

        else:
            raise ValueError(f'Unknown dataset {self.args.dataset}')

        self.trg_eval_loader = data.DataLoader(
                self.trg_parent_set, batch_size=self.args.batch_size,
                shuffle=False, pin_memory=True)


if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser()

    # main experiment parameters
    parser.add_argument("--model_arch", type=str, default='SegFormer-B0',
                        help="""Architecture name, see path_dicts.py
                            """)
    parser.add_argument("--src_dataset", type=str, default='Cityscapes',
                        help="Which source dataset to start from {Cityscapes}")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--root_exp_dir", type=str, default='results/debug/',
                        help="Where to save predictions.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")
    parser.add_argument("--samples_per_image", type=int, default=20000,
                        help="""Number of pixels per image to be chosen at random to evaluate calibration.
                        Note that using all pixels would lead to 2048*1024 = 2M pixels per image.""")
    

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
                        #{SYNTHIA, Cityscapes, FullCityscapes, ACDC, AROUND-1784-RL,
                        #1784, MTowerOffice_WiFi, MTowerOffice_AroundC}")
    parser.add_argument("--scene", type=str, default='aachen',
                        help="Scene, depends on specific datasets")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")

    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)

    # Full original image sizes
    if 'Cityscapes' == args.trg_dataset:
        args.input_size = '2048,1024'
    elif 'ACDC' in args.trg_dataset:
        args.input_size = '1920,1080' 
    elif 'IDD' in args.trg_dataset:
        args.input_size = '1280,720'
    elif args.trg_dataset.endswith(("fog", "night", "rain", "snow", "india", "all")):
        args.input_size = '512,512'
    else:
        raise NotImplementedError("Input size unknown")

    npr.seed(args.seed)

    solver_ops = SolverOps(args)

    print('Setting up experiment folder')
    solver_ops.setup_experiment_folder()

    print('Setting up data target loader')
    solver_ops.setup_target_data_loader()

    print(f'Building {args.model_arch} model')
    solver_ops.build_model()

    print('Start evaluating')
    solver_ops.extract_logits()

