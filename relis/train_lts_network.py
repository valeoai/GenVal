''' Adapted from https://github.com/uncbiag/LTS/blob/main/code/Tiramisu_calibration.py'''

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
import logging
import datetime


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

from lts_network import TemperatureModel

from dataset.dataset_mixed import MixedDatasets
# For mmseg models
from mmsegmentation.mmseg.apis import init_segmentor, inference_segmentor

from path_dicts import *


class SolverOps:

    def __init__(self, args):

        self.args = args
        self.args.num_classes = 19
        
        w_trg, h_trg = map(int, self.args.input_size.split(','))
        self.input_size = (w_trg, h_trg)
        
        self.initiate_logger()
        # Save arguments
        self.logger.info(self.args)

        
    def optimize_temperature(self):

        """
        Method to optimize temperatures for a model.
        All parameters setup by the user (args).
        """

        self.logger.info('Start training calib model')
        
        cudnn.enabled = True
        cudnn.deterministic = True

        nll_criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
        
        
        self.logger.info("Computing Loss")
        mean_val_loss = 0
        with torch.no_grad():
            for _, labels, image_path in self.val_loader:
                labels = labels.long().cuda()
                logits = inference_segmentor(self.model, image_path[0], output_logits=True, pre_softmax=True)
                        
                mean_val_loss += nll_criterion(logits, labels).item()
            mean_val_loss = mean_val_loss/len(self.val_loader)
        self.logger.info(f'Initial val loss: {mean_val_loss}')
        
        # Training settings
        max_epochs = self.args.epochs
        batch_size = self.args.batch_size
        lr = self.args.lr
        num_images = len(self.train_loader)
        
        # Calibration model
        temp_model = TemperatureModel(self.args.num_classes)
        temp_model.cuda()
        temp_model.weights_init()
        temp_model.train()
        
        optimizer = optim.Adam(temp_model.parameters(), lr=lr)
        optimizer.zero_grad()
        
        best_model = copy.deepcopy(temp_model)
        
        mean_val_loss = 1e6 # Initialize loss very high so first val iteration will surely become best model.
        idle_iteration_counter = 0
        self.logger.info("Begin training")
        for epoch in range(max_epochs):
            for i_iter, (image, labels, image_path) in enumerate(self.train_loader):
                with torch.no_grad():
                    logits = inference_segmentor(self.model, image_path[0], output_logits=True, pre_softmax=True)
                        
                image = image.cuda()
                labels = labels.long().cuda()
                scaled_logits = temp_model(logits, image)
                loss = nll_criterion(scaled_logits, labels)
                loss.backward()
                
                # Accumulate images in batches to do the weight update
                if ((i_iter + 1) % batch_size == 0) or (i_iter + 1 == num_images):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                ## save the current best model
                if (i_iter + 1) % (10 * batch_size) == 0 or (i_iter + 1 == num_images):
                    temp_model.eval()
                    with torch.no_grad():
                        tmp_loss = 0
                        for image, labels, image_path in self.val_loader:
                            labels = labels.long().cuda()
                            image = image.cuda()
                            logits = inference_segmentor(self.model, image_path[0], output_logits=True, pre_softmax=True)
                            scaled_logits = temp_model(logits, image)
                            tmp_loss += nll_criterion(scaled_logits, labels).item()
                        mean_tmp_loss = tmp_loss/len(self.val_loader)
                        self.logger.info(f'Epoch: {epoch} - iter: {i_iter // batch_size} - NLL: {mean_tmp_loss}')

                        if mean_tmp_loss < mean_val_loss:
                            mean_val_loss = mean_tmp_loss
                            self.logger.info('%d epoch, %d iteration current lowest - NLL: %.5f' % (epoch, i_iter // batch_size, mean_val_loss))
                            best_model = copy.deepcopy(temp_model)
                            # Reset idle iteration counter
                            idle_iteration_counter = 0
                        else:
                            idle_iteration_counter += 1
                            # Break computations because val NLL has not improved for a whole epoch
                            if num_images > 10*batch_size:
                                images_per_iter = 10*batch_size
                            else:
                                images_per_iter = num_images
                                
                            if idle_iteration_counter > 3*num_images/images_per_iter:
                                torch.save(best_model.state_dict(),
                                           os.path.join(self.args.results_dir,
                                                        'best_temperature_model_state_dict.pth'))
                                np.save(os.path.join(self.args.results_dir, 'best_nll.npy'),
                                        mean_val_loss)
                                self.logger.info('End of calibration. Early stopping')
                                
                                # Save also final model
                                torch.save(temp_model.state_dict(),
                                           os.path.join(self.args.results_dir,
                                                        'final_temperature_model_state_dict.pth'))
                                np.save(os.path.join(self.args.results_dir, 'final_nll.npy'),
                                        mean_tmp_loss)
                                self.logger.info('End of calibration. Early stopping')

                                with open(os.path.join(self.args.results_dir, self.args.DONE_name),'wb') as f:
                                    self.logger.info('Saving end of training file')
                                exit(0)
                            
                    temp_model.train()
                            
        torch.save(best_model.state_dict(), os.path.join(self.args.results_dir, 'best_temperature_model_state_dict.pth'))
        np.save(os.path.join(self.args.results_dir, 'best_nll.npy'), mean_val_loss)
        
        # Save final model
        torch.save(temp_model.state_dict(), os.path.join(self.args.results_dir, 'final_temperature_model_state_dict.pth'))
        np.save(os.path.join(self.args.results_dir, 'final_nll.npy'), mean_tmp_loss)
            
        self.logger.info('End of calibration.')

        with open(os.path.join(self.args.results_dir, self.args.DONE_name),'wb') as f:
            self.logger.info('Saving end of training file')

            
    def build_model(self):
        # Create network
        config = mmseg_models_configs[self.args.model_arch]
        checkpoint = mmseg_models_checkpoints[self.args.model_arch]
            
        from mmsegmentation.mmseg.apis import init_segmentor, inference_segmentor
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
        
        
    def setup_target_data_loader(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        self.logger.info('Setting up data target loader')
        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        dset_list = self.args.trg_dataset_list.split(',')
        
        full_dataset = MixedDatasets(scene_list, cond_list, dset_list)
        
        train_set, val_set = full_dataset.random_split_train_test(percentage_val=self.args.percentage_val,
                                                                  total_images=self.args.total_images)

        self.val_loader = data.DataLoader(
                val_set, batch_size=1,
                shuffle=True, pin_memory=True)
        
        self.train_loader = data.DataLoader(
                train_set, batch_size=1,
                shuffle=True, pin_memory=True)
        
    @staticmethod
    def pad_str(msg, total_len=70):
        rem_len = total_len - len(msg)
        return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\

    def initiate_logger(self):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(os.path.join(self.args.results_dir, 'log.txt'),'w'))
        logger.info(self.pad_str(' LOGISTICS '))
        logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
        logger.info('Output Name: {}'.format(self.args.results_dir))
        logger.info('User: {}'.format(os.getenv('USER')))
        self.logger = logger
        



if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser()

    # main experiment parameters
    parser.add_argument("--model_arch", type=str, default='SegFormer-B0',
                        help="Architecture name, see path_dicts.py")
    parser.add_argument("--src_dataset", type=str, default='Cityscapes',
                        help="Which source dataset to start from {Cityscapes}")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning Rate.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--total_images", type=int, default=500,
                        help="Max number of total images used to train the calib network.")
    parser.add_argument("--percentage_val", type=int, default=0.2,
                        help="Percentage of images used for validation.")
    
    

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to {Cityscapes, IDD, ACDC}")
    parser.add_argument("--trg_dataset_list", type=str, default='Cityscapes',
                        help="List of target datasets, one per cond and scene.")
    parser.add_argument("--scene", type=str, default='frankfurt',
                        help="Scene, depends on specific datasets")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")
    parser.add_argument("--results_dir", type=str, default='results/debug/model_temp_scaling/',
                        help="Where to save predictions.")


    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)

    npr.seed(args.seed)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    args.DONE_name = f'experiment.DONE'
        
        
    # Full original image sizes
    if 'Cityscapes' in args.trg_dataset:
        args.input_size = '2048,1024'
    elif 'ACDC' in args.trg_dataset:
        args.input_size = '1920,1080' 
    elif 'IDD' in args.trg_dataset:
        args.input_size = '1280,720'
    elif 'All' in args.trg_dataset: # If dataset is mixed use CS image size by default
        args.input_size = '2048,1024'
    else:
        raise NotImplementedError("Input size unknown")
        
    # check if experiment/testing was done already 
    if os.path.isfile(os.path.join(args.results_dir, args.DONE_name)) and not args.force_redo:
        print('DONE file present -- evaluation has already been carried out')
        print(os.path.join(args.results_dir, args.DONE_name))
        exit(0)
        
    solver_ops = SolverOps(args)
    
    
    solver_ops.setup_target_data_loader()

    
    solver_ops.build_model()

    
    solver_ops.optimize_temperature()

