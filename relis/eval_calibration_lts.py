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

from lts_network import TemperatureModel

# ours
from dataset.dataset_mixed import MixedDatasets
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD

from uncertainty_helpers import UncertaintyOps
from mmsegmentation.mmseg.apis import init_segmentor, inference_segmentor
from path_dicts import * # Import all paths to dsets, model checkpoints and configs

class SolverOps:

    def __init__(self, args):

        self.args = args
        self.args.num_classes = 19
        
        w_trg, h_trg = map(int, self.args.input_size.split(','))
        self.input_size = (w_trg, h_trg)

    def eval_calibration(self):

        """
        Method to evaluate calibration with temperatures scaled by calibration network.
        All parameters setup by the user (args).
        """
        
        logits_list, labels_list, indices_list, features_list, all_dsets_list = self.retrieve_feature_logits_labels_paths()
        
        scaled_logits, all_labels = self.create_labels_and_logits_stacks(logits_list, labels_list, indices_list, features_list, all_dsets_list)
        
        scaled_logits = scaled_logits.view(-1, self.args.num_classes)
        all_labels = all_labels.view(-1)
        
        # Logits to probs with temp scaling
        probs = UncertaintyOps.logits_to_probs(scaled_logits, T=1.0)
        
        # Compute calibration error
        print('Computing top1 ECE')
        top1_ece = UncertaintyOps.ECE(probs, all_labels, binning_strategy='equal_size', class_wise=False)
        print(top1_ece)
        
        print('Computing top1 Ada ECE')
        top1_ada_ece = UncertaintyOps.ECE(probs, all_labels, binning_strategy='equal_population', class_wise=False)
        print(top1_ada_ece)
        
        print('Computing KS error')
        ks_error = UncertaintyOps.ks_test(probs, all_labels, class_wise=False)
        print(ks_error)
        
        # Save results
        results = {
            'top1_ece': top1_ece,
            'top1_ada_ece': top1_ada_ece,
            'ks_error': ks_error,
        }
                                       
        # Save results
        results_filename = f'calib_metrics.pkl'
        
        with open(os.path.join(self.args.results_dir, results_filename),'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        print('End of evaluation.')

        with open(os.path.join(self.args.results_dir, self.args.DONE_name),'wb') as f:
            print('Saving end of training file')
        
        
        
    def retrieve_feature_logits_labels_paths(self):

        """
        Method to retrieve the paths to extracted features, logits and labels
        based on the condition and scene list.
        """

        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        dset_list = self.args.trg_dataset_list.split(',')
        
        features_sub_folder = f'extracted_features'
        logits_sub_folder = f'extracted_logits'
        model_arch_sub_folder = self.args.model_arch
        
        # Sorted (paired) preds and labels lists
        logits_list = []
        indices_list = []
        labels_list = []
        features_list = []
        all_dsets_list = []
        
        for scene, cond, dset in zip(scene_list, cond_list, dset_list):
            
            trg_sub_folder = f'{dset}_{scene}_{cond}'
            
            experiment_sub_folder = os.path.join(self.args.root_exp_dir, self.args.src_dataset,
                    model_arch_sub_folder, trg_sub_folder)
        
            self.logits_dir = os.path.join(experiment_sub_folder, logits_sub_folder,
                                           f'num_samples_{self.args.num_samples}')
            
            self.features_dir = os.path.join(experiment_sub_folder, features_sub_folder)
            
            # Unsorted all labels list
            all_labels = self.trg_parent_sets_dict[dset].annotation_path_list

            if cond == 'clean':
                cond = ''
            scene = f'/{scene}/'
            labels_list += sorted([x for x in all_labels if (scene in x and cond in x)])
            logits_list += sorted(glob.glob(os.path.join(self.logits_dir, '*logits.pt')))
            indices_list += sorted(glob.glob(os.path.join(self.logits_dir, '*indices.pt')))
            features_list += sorted(glob.glob(os.path.join(self.features_dir, '*_feat.pt')))
            # Copy dset for all logits in scene/cond
            all_dsets_list += [dset]*len(glob.glob(os.path.join(self.features_dir, '*_feat.pt')))
        
            
        return logits_list, labels_list, indices_list, features_list, all_dsets_list
    
    
    def create_labels_and_logits_stacks(self, logits_list, labels_list, indices_list, features_list, all_dsets_list):
        '''
        Retrieve all the labels and logits and store them in a tensor.
        '''
        
        cudnn.enabled = True
        cudnn.deterministic = True

        # Load temp model
        temp_model = TemperatureModel(self.args.num_classes)
        temp_model.load_state_dict(torch.load(self.args.calib_model_path))
        
        temp_model.eval()
        temp_model.cuda()
        
        all_logits = torch.zeros((len(logits_list), self.args.num_samples, self.args.num_classes))
        all_labels = torch.zeros((len(logits_list), self.args.num_samples))
        
        print("Extracting logits and labels")
        for i_iter, (image, labels, image_path) in enumerate(self.data_loader):
            with torch.no_grad():
                logits = inference_segmentor(self.model, image_path[0], output_logits=True, pre_softmax=True)
                    
                image = image.cuda()
                logits = temp_model(logits, image)
                
            logits = logits[0].cpu()
            labels = labels.cpu()
            
            indices = torch.load(indices_list[i_iter])
            
            # Filter labels
            labels = labels.view(-1)
            labels = labels[indices]
            
            # Filter logits
            logits = logits.permute(1,2,0).view(-1, self.args.num_classes)
            logits = logits[indices]
            
            # Stack logits and labels
            all_logits[i_iter] = logits.double() # For numerical precision
            all_labels[i_iter] = labels
            
        return all_logits, all_labels

            
    def build_model(self):
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

        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        dset_list = self.args.trg_dataset_list.split(',')
        
        dataset = MixedDatasets(scene_list, cond_list, dset_list)
        
        self.data_loader = data.DataLoader(
                dataset, batch_size=1,
                shuffle=False, pin_memory=True)
        
        self.setup_individual_datasets()
        
    
    def setup_individual_datasets(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        # (can also be a single environment)
        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        dset_list = self.args.trg_dataset_list.split(',')
        unique_dsets = list(set(dset_list))
            

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
                        batch_size=self.args.batch_size)

            elif dset=='IDD':
                trg_parent_set = IDD(
                        IDD_ROOT, dset_scenes, batch_size=self.args.batch_size)
            else:
                raise ValueError(f'Unknown dataset {self.args.dataset}')
            self.trg_parent_sets_dict[dset] = trg_parent_set
        


if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser(description="Extract features")

    # main experiment parameters
    parser.add_argument("--model_arch", type=str, default='SegFormer-B0',
                        help="""Architecture name, see path_dicts.py
                            """)
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
    parser.add_argument("--num_samples", type=int, default=20000,
                        help="Number of logits per image used to compute the calibration error.")
    parser.add_argument("--root_exp_dir", type=str, default='results/debug/',
                        help="Where to save predictions.")
    

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
    parser.add_argument("--trg_dataset_list", type=str, default='Cityscapes',
                        help="List of datasets per cond and scene.")
    parser.add_argument("--scene", type=str, default='frankfurt',
                        help="Scene, depends on specific datasets")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")
    parser.add_argument("--results_dir", type=str, default='results/debug/calib_model_temp_scaling/',
                        help="Where to save predictions.")
    parser.add_argument("--calib_model_path", type=str,
                        default='results/debug/model_temp_scaling/temperature_model_state_dict.pth',
                        help="Where to save predictions.")
    


    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)

    npr.seed(args.seed)
    
    # Full original image sizes
    if 'Cityscapes' in args.trg_dataset:
        args.input_size = '2048,1024'
    elif 'ACDC' in args.trg_dataset:
        args.input_size = '1920,1080' 
    elif 'IDD' in args.trg_dataset:
        args.input_size = '1280,720'
    else:
        raise NotImplementedError("Input size unknown")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    args.DONE_name = f'experiment.DONE'
    
    # check if experiment/testing was done already 
    if os.path.isfile(os.path.join(args.results_dir, args.DONE_name)) and not args.force_redo:
        print('DONE file present -- evaluation has already been carried out')
        print(os.path.join(args.results_dir, args.DONE_name))
        exit(0)
        
    solver_ops = SolverOps(args)

    print('Setting up data target loader')
    solver_ops.setup_target_data_loader()

    print(f'Building {args.model_arch} model')
    solver_ops.build_model()

    print('Start evaluating')
    solver_ops.eval_calibration()

