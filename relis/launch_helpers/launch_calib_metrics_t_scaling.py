'''
Helper script to create several launch commands that evaluate model calibration
on different datasets after temperature scaling.
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

for dset in ['CS', 'ACDC', 'IDD']:
    
    trg_dataset_list, scene_list, cond_list = [], [], []

    if dset == 'CS':
        run_exps_helpers.update_cityscapes_lists('test-1-clean', trg_dataset_list, scene_list, cond_list)
    elif dset == 'ACDC':
        run_exps_helpers.update_acdc_lists('train', trg_dataset_list, scene_list, cond_list)
    elif dset == 'IDD':
        run_exps_helpers.update_idd_lists('train_clean', trg_dataset_list, scene_list, cond_list)
        
        
    # Which models we want to use. Names should correpond to the keys in path_dicts.py
    model_arch_list = [
                # 'SETR-PUP',
                # 'SETR-MLA',
                # 'SETR-Naive',
                'SegFormer-B0',
                # 'SegFormer-B1',
                # 'SegFormer-B2',
                # 'SegFormer-B3',
                # 'SegFormer-B4',
                # 'SegFormer-B5',
                # 'DLV3+ResNet50',
                # 'DLV3+ResNet101',
                # 'DLV3+ResNet18',
                # 'Segmenter',
                # 'ConvNext',
                # 'UpperNetR18',
                # 'UpperNetR50',
                # 'UpperNetR101',
                # 'ConvNext-B-In1K',
                # 'ConvNext-B-In21K',
                # 'BiT-R50x1-In1K',
                # 'BiT-R50x1-In21K',
                # 'Mask2Former',
                # 'MaskFormer',
                # 'SwinLarge',
                # 'SegFormer-B5_v2',
                # 'SegFormer-B5_v3',
                # 'Segmenter_short',
                ]
    
    # Num samples per image to estimate calibration error
    nsamples = 20000

    if len(set(trg_dataset_list)) == 1: 
        trg_dataset_ = trg_dataset_list[0]
    else:
        trg_dataset_ = 'All_dsets'

    # Format scenes and conditions
    scenes = ''
    conditions = ''
    trg_datasets = ''
    for scene, cond, dset in zip(scene_list, cond_list, trg_dataset_list):
        scenes = scenes + scene + ','
        conditions = conditions + cond + ','
        trg_datasets = trg_datasets + dset + ','

    scenes = scenes[:-1]
    conditions = conditions[:-1]
    trg_datasets = trg_datasets[:-1]
    
    #     number_of_clusters = [1, 4, 8, 16, 32]
    number_of_clusters = [1]
    
    def get_results_dir(trg_dataset_, model_arch_, root_exp_dir_, n_clusters_, val_dset, clust_name, cluster_assignment):
        if cluster_assignment == 'soft':
            clust_name += 'soft_assignment_'
        results_dir = os.path.join(f'{root_exp_dir_}Cityscapes/{model_arch_}',
                                   f'calib_metrics_clusters/{val_dset}/',
                                   f'{n_clusters}_{clust_name}clusters/{trg_dataset_}/')
        return results_dir

    if trg_dataset_ == 'Cityscapes':
        val_dsets = [trg_dataset_, 'All_dsets', 'Cityscapes_augmented']
    elif trg_dataset_ == 'IDD':
         val_dsets = [trg_dataset_, 'All_dsets', 'Cityscapes', 'Cityscapes_augmented']
    elif trg_dataset_ == 'ACDC':
         val_dsets = [trg_dataset_, 'All_dsets', 'Cityscapes', 'Cityscapes_augmented']

    # val_dsets = [trg_dataset_]
    
    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'
    
    cluster_assignment = ''
    for clustering_method_ in ['kmeans']:
        if clustering_method_ == 'kmeans':
            clust_name = ''
        else:
            clust_name = clustering_method_ + '_'
        for model_arch_ in model_arch_list:
            for n_clusters in number_of_clusters:
                for val_dset in val_dsets:
                    cluster_and_temp_path = os.path.join(f'{root_exp_dir_}Cityscapes/',
                                                         f'{model_arch_}/clusters_and_temp/{val_dset}/',
                                                         f'{n_clusters}_{clust_name}clusters/')
                    cluster_model_path_ = os.path.join(cluster_and_temp_path, 'cluster_model.joblib')
                    temperatures_path_ = os.path.join(cluster_and_temp_path, 'best_temperatures.npy')

                    # DONE filename (check if experiment has already been done)
                    results_dir = get_results_dir(trg_dataset_, model_arch_,
                                                      root_exp_dir_, n_clusters, val_dset, clust_name, cluster_assignment)
                    done_filename = os.path.join(results_dir, 'experiment.DONE')
                    # Check if done file is present 
                    if os.path.isfile(done_filename) and not force_redo_:
                        pass
                    else:
                        print(f'python -u eval_calibration_clustering.py'+
                                  f' --force_redo={force_redo_}' +
                                  f' --trg_dataset={trg_dataset_}' +
                                  f' --trg_dataset_list={trg_datasets}'
                                  f' --scene={scenes}' +
                                  f' --cond={conditions}' +
                                  f' --model_arch={model_arch_}' +
                                  f' --num_samples={nsamples}'+
                                  f' --cluster_assignment={cluster_assignment}'+
                                  f' --cluster_model_path={cluster_model_path_}'+
                                  f' --temperatures_path={temperatures_path_}'+
                                  f' --root_exp_dir={root_exp_dir_}'+
                                  f' --results_dir={results_dir}')