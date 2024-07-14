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
from uncertainty_helpers import UncertaintyOps

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

        self.image_ops = ImageOps()

        w_trg, h_trg = map(int, self.args.input_size.split(','))
        self.input_size = (h_trg, w_trg)

    @staticmethod
    def label_mapping(input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)

    def eval_calibration(self):

        """
        Evaluate calibration performance.
        All parameters setup by the user (args).
        """

        IDD_TO_CITYSCAPES_MAPPING = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                                     5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}
        # Varied info about cityscapes dataset
        with open('./dataset/cityscapes_list/info.json', 'r') as f:
            info = json.load(f)
        # Particularly, we need the label mapping info
        mapping = np.array(info['label2train'], dtype=np.int64)
    
        # Logits and indices paths
        (labels_list, logits_list,
         indices_list, dset_list) = self.retrieve_sorted_labels_logits_and_indices_paths()
        
        # Check number of logit samples is correct
        assert len(torch.load(indices_list[0])) == self.args.num_samples
        assert (len(labels_list) == len(indices_list) == len(logits_list) == len(dset_list))
        
        all_logits = torch.zeros((len(logits_list), self.args.num_samples, self.args.num_classes))
        all_labels = torch.zeros((len(logits_list), self.args.num_samples))
                
        for i_iter, (logits, indices, labels, dset) in enumerate(zip(logits_list, indices_list, labels_list, dset_list)):
            
            # Load logits and labels
            logits = torch.load(logits)
            indices = torch.load(indices)
            label_image = Image.open(labels)
            labels = np.array(label_image)
            
            if ('Cityscapes' in dset) or ('ACDC' in dset):
                # Labels in cityscapes dset need to be mapped to the 19 classes since they contain more information
                aux_labels = self.label_mapping(labels, mapping)
            elif 'IDD' in dset:
                aux_labels = 255 * np.ones(labels.shape, dtype=np.float32)
                for k, v in IDD_TO_CITYSCAPES_MAPPING.items():
                    aux_labels[labels == k] = v
                    
            labels = torch.tensor(aux_labels)
            
            # Filter labels
            labels = labels.view(-1)
            labels = labels[indices]
            
            # Stack logits and labels
            all_logits[i_iter] = torch.tensor(logits.astype('float64')) # For numerical precision
            all_labels[i_iter] = labels
            
        # Reshape logits and labels
        all_logits = all_logits.view(-1, self.args.num_classes)
        all_labels = all_labels.view(-1)
        
        # Logits to probs with temp scaling
        if 'Mask' in self.args.model_arch:
            probs = all_logits
        else:
            probs = UncertaintyOps.logits_to_probs(all_logits, T=args.temp)
            
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


    def retrieve_sorted_labels_logits_and_indices_paths(self):

        """
        Method to retrieve the paths to predicted images based on the condition and scene list.
        """

        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        dset_list = self.args.trg_dataset_list.split(',')
        
        method_sub_folder = f'extracted_logits'
        model_arch_sub_folder = self.args.model_arch
        
        # Sorted (paired) dset, preds and labels lists
        logits_list = []
        indices_list = []
        labels_list = []
        all_dsets_list = []
        
        for scene, cond, dset in zip(scene_list, cond_list, dset_list):
            trg_sub_folder = f'{dset}_{scene}_{cond}'
            
            self.model_dir = os.path.join(
                    self.args.root_exp_dir, self.args.src_dataset,
                    model_arch_sub_folder, trg_sub_folder, method_sub_folder)

            self.logits_dir = os.path.join(
                    self.model_dir, f'num_samples_{self.args.num_samples}')
            
            # Unsorted all labels list
            all_labels = self.trg_parent_sets_dict[dset].annotation_path_list

            if cond == 'clean':
                cond = ''
            scene = f'/{scene}/'
            labels_list += sorted([x for x in all_labels if (scene in x and cond in x)])
            logits_list += sorted(glob.glob(os.path.join(self.logits_dir, '*logits.pt')))
            indices_list += sorted(glob.glob(os.path.join(self.logits_dir, '*indices.pt')))
            # Copy dset for all logits in scene/cond
            all_dsets_list += [dset]*len(glob.glob(os.path.join(self.logits_dir, '*logits.pt')))
        
        return labels_list, logits_list, indices_list, all_dsets_list
    

    def setup_target_data_loader(self):

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
            
            elif dset=='Cityscapes_fog':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_FOG,
                        scene_list, cond_list)
        
            elif dset=='Cityscapes_night':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_NIGHT,
                        scene_list, cond_list)

            elif dset=='Cityscapes_rain':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_RAIN,
                        scene_list, cond_list)

            elif dset=='Cityscapes_snow':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_SNOW,
                        scene_list, cond_list)

            elif dset=='Cityscapes_india':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_INDIA,
                        scene_list, cond_list)

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
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    # main experiment parameters
    parser.add_argument("--model_arch", type=str, default='SegFormer-B0',
                        help="""Architecture name, see path_dicts.py
                            """)
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Parameter for logits temperature scaling.")
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
    parser.add_argument("--num_samples", type=int, default=20000,
                        help="""Number of pixels per image to be chosen at random to evaluate calibration.
                        Note that using all pixels would lead to 2048*1024 = 21M pixels per image.
                        """)
    parser.add_argument("--results_dir", type=str, default='results/debug/debug_calib/',
                        help="Where to save predictions.")

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
    parser.add_argument("--trg_dataset_list", type=str, default='Cityscapes',
                        help="List of datasets per cond and scene.")
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
    elif args.trg_dataset.endswith(("fog", "night", "rain", "snow", "india")):
        args.input_size = '512,512'
    else:
        raise NotImplementedError("Input size unknown")

    npr.seed(args.seed)
    
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

    print('Start evaluating')
    solver_ops.eval_calibration()

