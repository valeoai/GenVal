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
from torch.utils import data

import numpy as np
import numpy.random as npr

from PIL import Image

# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD

from clustering_helpers import ClusteringOPS
from uncertainty_helpers import UncertaintyOps

from path_dicts import * # Import all paths to dsets, model checkpoints and configs

class SolverOps:

    def __init__(self, args):
        self.args = args
        if 'Cityscapes' in self.args.src_dataset:
            self.args.num_classes = 19
        else:
            self.args.num_classes = 19
            # raise NotImplementedError()
        
        

    def cluster_features(self):

        """
        Apply clustering method to features.
        All parameters setup by the user (args).
        """

        print('Extracting feature paths')
        # Feature paths
        (logits_list, labels_list,
         indices_list, features_list, dset_list) = self.retrieve_feature_logits_labels_paths()
        
        all_features = self.create_feature_stack(features_list)
        
        all_logits, all_labels = self.create_labels_and_logits_stacks(logits_list, labels_list, indices_list, dset_list)
            
        if self.args.n_clusters == -1:
            self.args.n_clusters = len(features_list)
        
        # Compute clustering
        clustering = ClusteringOPS(clustering_method=self.args.clustering_method,
                                  n_clusters=self.args.n_clusters, seed=self.args.seed,
                                  )
        
        
        uncertainty_ops = UncertaintyOps()
        
        # Cluster elements
        cluster_labels = clustering.fit_predict(all_features)
        
        # Save clustering method
        model_filename = os.path.join(self.args.results_dir, 'cluster_model.joblib')
        clustering.save_model(model_filename)
        print('Clustering done. Computing best temperatures per cluster...')
        
        # Option not to use cuda if we get gpu memory error with only one cluster
        if self.args.n_clusters >= 1:
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = False
        
        # Compute best temperature per cluster
        if self.args.per_class_temperature == 1:
            best_temperatures = np.zeros((self.args.n_clusters, self.args.num_classes))
        else:
            best_temperatures = np.zeros(self.args.n_clusters)
            
        for ii, selected_label in enumerate(range(self.args.n_clusters)):
            selected_logits, selected_labels = self.select_logits_labels_from_cluster(all_logits,
                                                                                      all_labels,
                                                                                      cluster_labels,
                                                                                      selected_label)
            print(f'Cluster {ii} of size {selected_logits.shape[0]}')
            if self.args.per_class_temperature == 1:
                # In this case each best temperatures will be a matrix
                best_temperatures[ii] = uncertainty_ops.find_best_temperature_per_class_grid_search(
                                                                                          selected_logits,
                                                                                          selected_labels,
                                                                                          self.args.num_classes,
                                                                                          min_t=self.args.min_t,
                                                                                          max_t=self.args.max_t,
                                                                                          step=self.args.step,
                                                                                          cuda=use_cuda,
                                                                                          device=self.args.gpu)
            else:
                best_temperatures[ii] = uncertainty_ops.find_best_temperature_grid_search(
                                                                                          selected_logits,
                                                                                          selected_labels,
                                                                                          min_t=self.args.min_t,
                                                                                          max_t=self.args.max_t,
                                                                                          step=self.args.step,
                                                                                          cuda=use_cuda,
                                                                                          device=self.args.gpu)
        
        
        # Save results
        temperatures_filename = os.path.join(self.args.results_dir, 'best_temperatures.npy')
        np.save(temperatures_filename, best_temperatures)
        
        print('Saved best temperatures.')

        with open(os.path.join(self.args.results_dir, self.args.DONE_name),'wb') as f:
            print('Saving end of training file.')
            
            
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
    
    def reshape_logits_labels(self, logits, labels):
        logits = logits.view(-1, self.args.num_classes)
        labels = labels.view(-1)
        return logits, labels

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
        mapping = np.array(info['label2train'], dtype=np.int64)
    
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
    
    def select_logits_labels_from_cluster(self, all_logits, all_labels, cluster_labels, selected_label):
        
        '''
        Given a cluster label, select all those logits and pixel labels that belong to that cluster.
        '''
        
        all_logits = all_logits[cluster_labels==selected_label]
        all_labels = all_labels[cluster_labels==selected_label]
        
        # Reshape logits and labels
        all_logits = all_logits.view(-1, self.args.num_classes)
        all_labels = all_labels.view(-1)
        
        return all_logits, all_labels
        
        
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
            
            experiment_logits_sub_folder = os.path.join(self.args.logits_dir, self.args.src_dataset,
                    model_arch_sub_folder, trg_sub_folder)
            
            experiment_features_sub_folder = os.path.join(self.args.features_dir, self.args.src_dataset,
                    model_arch_sub_folder, trg_sub_folder)

        
            self.logits_dir = os.path.join(experiment_logits_sub_folder, logits_sub_folder,
                                           f'num_samples_{self.args.num_samples}')
            
            self.features_dir = os.path.join(experiment_features_sub_folder, features_sub_folder)
                        
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

        # To use logits and features from bigger datasets, to try several runs on smaller sets for variance computation
        if self.args.subsamples is not None:
            # Generate subsamples random idxs
            idxs = random.sample(range(0, len(all_dsets_list)), self.args.subsamples)

            # Select subsamples
            labels_list = [labels_list[i] for i in idxs]
            logits_list = [logits_list[i] for i in idxs]
            indices_list = [indices_list[i] for i in idxs]
            features_list = [features_list[i] for i in idxs]
            all_dsets_list = [all_dsets_list[i] for i in idxs]
        
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

            elif dset=='Cityscapes_all':
                trg_parent_set = Cityscapes(
                        CITYSCAPES_ALL,
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
    parser.add_argument("--src_dataset", type=str, default='Cityscapes',
                        help="Which source dataset to start from {Cityscapes}")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--features_dir", type=str, default="results/features")
    parser.add_argument("--logits_dir", type=str, default="results/logits")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")
    parser.add_argument("--results_dir", type=str, default='results/debug/debug_clusters/',
                        help="Where to save predictions.")
    parser.add_argument("--num_samples", type=int, default=20000,
                        help="""Number of pixels per image to be chosen at random to evaluate calibration.
                        Note that using all pixels would lead to 2048*1024 = 21M pixels per image.
                        """)
    # Clustering
    parser.add_argument("--clustering_method", type=str, default='kmeans',
                        help="Clustering method to use {'kmeans', TODO}")
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Number of clusters")
    parser.add_argument("--per_class_temperature", type=int, default=0,
                        help="""If set to 1, within each cluster we will compute
                        the best temperature per class independently""")
    
    # Temp scaling optimization
    parser.add_argument("--calib_metric", type=str, default='ece',
                        help="Metric to optimize to find best temperature. One of {'ece', 'nll'}.")
    parser.add_argument("--min_t", type=float, default=0.5,
                        help="Min temperature for grid search")
    parser.add_argument("--max_t", type=float, default=10,
                        help="Max temperature for grid search")
    parser.add_argument("--step", type=float, default=0.01,
                        help="Step size of temperature for grid search")
    
    

    # for target
    parser.add_argument("--trg_dataset_list", type=str, default='Cityscapes',
                        help="List of dsets per scene and condition")
    parser.add_argument("--scene", type=str, default='lindau',
                        help="Scene, depends on specific datasets")
    parser.add_argument("--cond", type=str, default='clean',
                        help="Condition, depends on specific datasets")

    # Subsampling examples for variance computation on multiple calibrations
    parser.add_argument("--subsamples", type=int, default=None,
                        help="Number of samples from the whole calibration set, sampled at random, to use during best temperature computation")

    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)

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

    print('Cluster features')
    solver_ops.cluster_features()

