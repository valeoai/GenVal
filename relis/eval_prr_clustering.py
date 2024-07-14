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

from uncertainty_helpers import UncertaintyOps
from clustering_helpers import ClusteringOPS

from path_dicts import * # Import all paths to dsets, model checkpoints and configs


class SolverOps:

    def __init__(self, args):

        self.args = args
        self.args.num_classes = 19

    @staticmethod
    def label_mapping(input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)

    def eval_prr(self):

        """
        Evaluate Prediction Rejection Ratio (PRR) performance.
        All parameters setup by the user (args).
        """

        # Logits and indices paths
        (logits_list, labels_list,
         indices_list, features_list, dset_list) = self.retrieve_feature_logits_labels_paths()
        
        # Check number of logit samples is correct
        assert len(torch.load(indices_list[0])) == self.args.num_samples
        
        all_features = self.create_feature_stack(features_list)
        all_logits, all_labels = self.create_labels_and_logits_stacks(logits_list, labels_list, indices_list, dset_list)
        
        # Compute clustering
        clustering = ClusteringOPS()
        clustering.load_model(self.args.cluster_model_path)
        print(f'Loaded clustering from {self.args.cluster_model_path}.')
        
        best_temperatures = np.load(self.args.temperatures_path)
        print(f'Loaded temperatures from {self.args.temperatures_path}.')
        
        if '_gmm_clusters' in self.args.cluster_model_path:
            assert len(best_temperatures) == clustering.cluster.means_.shape[0]
        else: # kmeans
            assert len(best_temperatures) == clustering.cluster.cluster_centers_.shape[0]
            
        print(f'Assigning clusters to images.')
        # Cluster elements
        if self.args.cluster_assignment == 'soft':
            cluster_probs = clustering.predict_proba(all_features)
        else:
            cluster_labels = clustering.predict_cluster(all_features)
        
        print(f'Scaling logits according to cluster temperatures.')
        # Compute temperature scaled logits
        if self.args.cluster_assignment == 'soft':
            all_logits = self.scaled_logits_soft_assignment(all_logits, cluster_probs, best_temperatures)
        else:
            all_logits = self.scaled_logits(all_logits, cluster_labels, best_temperatures)
            
        # Reshape logits and labels
        all_logits = all_logits.view(-1, self.args.num_classes)
        all_labels = all_labels.view(-1)
        
        # Compute PRR
        print('Computing PRR metric')
        uncertainty_ops = UncertaintyOps()
        prr = uncertainty_ops.prediction_rejection_ratio(all_labels, all_logits, self.args.confidence_metric)
                                       
        # Save results
        results_filename = f'prediction_rejection_ratio.npy'
        
        print('Saving results')
        
        np.save(os.path.join(self.args.results_dir, results_filename), prr)

        print('End of evaluation.')

        with open(os.path.join(self.args.results_dir, self.args.DONE_name),'wb') as f:
            print('Saving end of training file')


    def create_feature_stack(self, features_list):
        
        # Check feature dimensions
        self.args.feature_dim = torch.load(features_list[0]).shape[0]
        
        print(f'Found {len(features_list)} features of {self.args.feature_dim} dimensions')
        
        all_features = torch.zeros((len(features_list), self.args.feature_dim))
                
        for i_iter, feature_path in enumerate(features_list):
            
            # Load features
            features = torch.load(feature_path)
            
            # Stack logits and labels
            all_features[i_iter] = features.to(torch.double) # For numerical precision
        
        return all_features
    
    @staticmethod
    def _label_mapping(input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)
        
    def create_labels_and_logits_stacks(self, logits_list, labels_list, indices_list, dset_list):
        '''
        Retrieve all the labels and logits from the list and store them in a tensor.
        Also process the labels so they all match with the same classes.
        '''
        
        IDD_TO_CITYSCAPES_MAPPING = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                                     5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}
        # Varied info about cityscapes dataset
        with open('./dataset/cityscapes_list/info.json', 'r') as f:
            info = json.load(f)
        # Particularly, we need the label mapping info
        mapping = np.array(info['label2train'], dtype=np.int)
    
        # Check number of logit samples is correct
        assert len(torch.load(indices_list[0])) == self.args.num_samples
        
        all_logits = torch.zeros((len(logits_list), self.args.num_samples, self.args.num_classes))
        all_labels = torch.zeros((len(logits_list), self.args.num_samples))
        
        for i_iter, (logits, labels, indices, dset) in enumerate(zip(logits_list, labels_list, indices_list, dset_list)):
            
            # Load logits and labels
            logits = torch.load(logits)
            indices = torch.load(indices)
            label_image = Image.open(labels)
            labels = np.array(label_image)
            
            if ('Cityscapes' in dset) or ('ACDC' in dset):
                # Labels in cityscapes dset need to be mapped to the 19 classes since they contain more information
                aux_labels = self._label_mapping(labels, mapping)
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
            
        return all_logits, all_labels
    
    
    @staticmethod
    def scaled_logits(all_logits, cluster_labels, best_temperatures):
        
        '''
        Scale logits according to best temperature per cluster.
        '''
        
        scaled_logits = torch.zeros_like(all_logits).double()
        if best_temperatures.shape == 2:
            predicted_class = torch.argmax(all_logits, dim=-1)
            for ii, T in enumerate(best_temperatures):
                cluster_index = cluster_labels == ii
                for jj, t in enumerate(T):
                    index_class = predicted_class[cluster_index] == jj
                    scaled_logits[cluster_index][index_class] = all_logits[cluster_index][index_class] / t
        else:
            for ii, T in enumerate(best_temperatures):
                index = cluster_labels == ii
                scaled_logits[index] = (all_logits[index] / T).double()
        
        return scaled_logits
    
    @staticmethod
    def scaled_logits_soft_assignment(all_logits, cluster_probs, best_temperatures):
        
        '''
        Scale logits according to best temperature per cluster.
        '''
        scaled_logits = torch.zeros_like(all_logits).double()
        if best_temperatures.shape == 2:
            predicted_class = torch.argmax(all_logits, dim=-1)
            for jj in range(best_temperatures.shape[2]):
                T = best_temperatures[:, jj]
                soft_temperatures = (cluster_probs * T).sum(axis=1)
                index = predicted_class == jj
                soft_temperatures = soft_temperatures[index[0]]
                
                scaled_logits[index] = all_logits[index] / soft_temperatures[:, np.newaxis]
        else:
            soft_temperatures = (cluster_probs * best_temperatures).sum(axis=1)
            scaled_logits = all_logits / soft_temperatures[:, np.newaxis, np.newaxis]
        
        return scaled_logits
    

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
            
            if self.args.trg_dataset=='Cityscapes':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_ROOT,
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
    parser = argparse.ArgumentParser()

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
                        help="""Number of pixels per image to be chosen at random to evaluate PRR.
                        Note that using all pixels would lead to 2048*1024 = 21M pixels per image.
                        """)
    parser.add_argument("--results_dir", type=str, default='results/debug/debug_prr_clustering/',
                        help="Where to save predictions.")
    # For prediction rejection
    parser.add_argument("--confidence_metric", type=str, default='prob',
                        help="""Which confidence score to use for OOD detection:
                            -prob: Probability of the predicted class.
                            -entropy: Logits entropy.
                        """)

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
    parser.add_argument("--trg_dataset_list", type=str, default='Cityscapes',
                        help="List of datasets per cond and scene.")
    parser.add_argument("--scene", type=str, default='aachen',
                        help="Scene, depends on specific datasets")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")

    # Clustering
    parser.add_argument("--cluster_model_path", type=str, 
                        default='results/debug/debug_clusters/cluster_model.joblib',
                        help="Path to retrieve the pre-trained clustering model")
    
    # Temp scaling optimization
    parser.add_argument("--temperatures_path", type=str, 
                        default='results/debug/debug_clusters/best_temperatures.npy',
                        help="Path to retrieve the pre-trained clustering model")
    parser.add_argument("--cluster_assignment", type=str, 
                        default='hard',
                        help="""Cluster assignment: 'hard' performs exclusive cluster assignment
                                                    'soft' performs probabilistic cluster assignment and temperatures are averaged.""")


    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)


    # Full original image sizes
    if 'Cityscapes' in args.trg_dataset:
        args.input_size = '2048,1024'
    elif 'ACDC' in args.trg_dataset:
        args.input_size = '1920,1080' 
    elif 'IDD' in args.trg_dataset:
        args.input_size = '1280,720'
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
    solver_ops.eval_prr()

