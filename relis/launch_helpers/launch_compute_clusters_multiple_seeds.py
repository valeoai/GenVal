'''
Helper script to create several launch commands to compute clusters of 
images based on the model's backbone features and compute optimal temperature
for calibration within each cluster of images.
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

datasets = ['CS', 'ACDC', 'IDD', 'CS_augmented', 'All']
for dset in datasets:
    
    trg_dataset_list, scene_list, cond_list = [], [], []

    if dset == 'CS':
        run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
    elif dset == 'CS_augmented':
        run_exps_helpers.update_cityscapes_lists('val-1-augmented', trg_dataset_list, scene_list, cond_list)
    elif dset == 'ACDC':
        run_exps_helpers.update_acdc_lists('val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'IDD':
        run_exps_helpers.update_idd_lists('val_clean', trg_dataset_list, scene_list, cond_list)
    elif dset == 'All':
        run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_idd_lists('val_clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_acdc_lists('val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'CS_ACDC':
        run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_acdc_lists('val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'CS_IDD':
        run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_idd_lists('val_clean', trg_dataset_list, scene_list, cond_list)

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

    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'

    if len(set(trg_dataset_list)) == 1:
        trg_dataset_ = trg_dataset_list[0]
    elif len(set(trg_dataset_list)) == 2:
        trg_dataset_ = dset
    else:
        trg_dataset_ = 'All_dsets'
    if 'augmented' in dset:
        trg_dataset_ += '_augmented'


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

    for clustering_method_ in ['kmeans']:
        if clustering_method_ == 'kmeans':
            clust_name = ''
        elif clustering_method_ == 'gmm':
            clust_name = clustering_method_ + '_reg_'

        def get_done_filename(trg_dataset_, model_arch_, root_exp_dir_, n_clusters_, clust_name, seed):
            # DONE filename (check if experiment has already been done)
            results_dir = f'{root_exp_dir_}Cityscapes/{model_arch_}/clusters_and_temp/{trg_dataset_}/{n_clusters_}_{clust_name}clusters_seed_{seed}/'
            DONE_name = f'experiment.DONE'
            return os.path.join(results_dir, DONE_name)

        # for seed in [42, 43, 44, 45, 46]:
        for seed in [42, 43]:
            for model_arch_ in model_arch_list:
                # for n_clusters_ in [4, 8, 16, 32]:
                for n_clusters_ in [8]:
                    # DONE filename (check if experiment has already been done)
                    done_filename = get_done_filename(trg_dataset_, model_arch_,
                                                      root_exp_dir_, n_clusters_,
                                                      clust_name, seed)
                    # Check if done file is present 
                    if os.path.isfile(done_filename) and not force_redo_:
                        pass
                    else:
                        print(f'python -u compute_clusters_temp_scaling.py'+
                                  f' --force_redo={force_redo_}' +
                                  f' --trg_dataset_list={trg_datasets}' +
                                  f' --scene={scenes}' +
                                  f' --cond={conditions}' +
                                  f' --seed={seed}' +
                                  f' --model_arch={model_arch_}' +
                                  f' --n_clusters={n_clusters_}' +
                                  f' --clustering_method={clustering_method_}' +
                                  f' --root_exp_dir={root_exp_dir_}'+
                                  f' --results_dir={root_exp_dir_}Cityscapes/{model_arch_}/'+
                                  f'clusters_and_temp/{trg_dataset_}/{n_clusters_}_{clust_name}clusters_seed_{seed}/')
