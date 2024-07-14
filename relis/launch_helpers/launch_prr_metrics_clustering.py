'''
Helper script to create several launch commands that evaluate the misclassification error
(Prediction rejection ratio) for different datasets after temp scaling via clustering.
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

    trg_dataset_ = trg_dataset_list[0]

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
    
    def get_results_dir(trg_dataset_, model_arch_, root_exp_dir_, n_clusters_, val_dset, clust_name, cluster_assignment, seed, metric):
        if cluster_assignment == 'soft':
            clust_name += 'soft_assignment_'
        if metric == 'entropy':
            metric = '_entropy'
        else:
            metric = ''
        results_dir = os.path.join(f'{root_exp_dir_}Cityscapes/{model_arch_}',
                                   f'prr_metrics_clustering{metric}/{val_dset}/',
                                   f'{n_clusters}_{clust_name}clusters_seed_{seed}/{trg_dataset_}/'
                                   )
        return results_dir

    val_dsets = ['Cityscapes']
    
    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'
    
    cluster_assignment = ''
    for clustering_method_ in ['kmeans']:
        if clustering_method_ == 'kmeans':
            clust_name = ''
        else:
            clust_name = clustering_method_ + '_'
        
        # for seed in [42, 43, 44, 45, 46]:
        for seed in [42]:
            for model_arch_ in model_arch_list:
                # for n_clusters in [16]:
                for n_clusters in [8]:
                    for val_dset in val_dsets:
                        for metric in ['prob', 'entropy']:
                            cluster_and_temp_path = os.path.join(f'{root_exp_dir_}Cityscapes/',
                                                                 f'{model_arch_}/clusters_and_temp/{val_dset}/',
                                                                 f'{n_clusters}_{clust_name}clusters_seed_{seed}/')
                            cluster_model_path_ = os.path.join(cluster_and_temp_path, 'cluster_model.joblib')
                            temperatures_path_ = os.path.join(cluster_and_temp_path, 'best_temperatures.npy')

                            # DONE filename (check if experiment has already been done)
                            results_dir = get_results_dir(trg_dataset_, model_arch_,
                                                          root_exp_dir_, n_clusters, val_dset,
                                                          clust_name, cluster_assignment, seed, metric)
                            done_filename = os.path.join(results_dir, 'experiment.DONE')
                            # Check if done file is present 
                            if os.path.isfile(done_filename) and not force_redo_:
                                pass
                            else:
                                print(f'python -u eval_prr_clustering.py' +
                                          f' --force_redo={force_redo_}' +
                                          f' --trg_dataset={trg_dataset_}' +
                                          f' --trg_dataset_list={trg_datasets}' +
                                          f' --scene={scenes}' +
                                          f' --cond={conditions}' +
                                          f' --model_arch={model_arch_}' +
                                          f' --num_samples={nsamples}' +
                                          f' --cluster_assignment={cluster_assignment}' +
                                          f' --cluster_model_path={cluster_model_path_}' +
                                          f' --temperatures_path={temperatures_path_}' +
                                          f' --results_dir={results_dir}' +
                                          f' --confidence_metric={metric}')