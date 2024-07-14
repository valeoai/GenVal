'''
Helper script to create several launch commands that evaluate the accuracy and miou
on different datasets. Require first to launch eval.py 
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0 # Set to 1 to overwrite results. Otherwise it skips done experiments.

for dset in ['CS', 'ACDC', 'IDD']:
    
    trg_dataset_list, scene_list, cond_list = [], [], []

    # Note for IDD and ACDC we are using train partition as test set (since they do not have one)
    # For Cityscapes we use a partition of the original val set as test (since models are trained already on CS).

    if dset == 'CS':
        run_exps_helpers.update_cityscapes_lists('test-1-clean', trg_dataset_list, scene_list, cond_list)
    elif dset == 'ACDC':
        run_exps_helpers.update_acdc_lists('train', trg_dataset_list, scene_list, cond_list)
    elif dset == 'IDD':
        run_exps_helpers.update_idd_lists('train_clean', trg_dataset_list, scene_list, cond_list)
    elif dset == 'All':
        run_exps_helpers.update_cityscapes_lists('test-1-clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_idd_lists('train_clean', trg_dataset_list, scene_list, cond_list)
        run_exps_helpers.update_acdc_lists('train', trg_dataset_list, scene_list, cond_list)

    print(trg_dataset_list)

    # Which models we want to use. Names should correpond to the keys in path_dicts.py
    model_arch_list = [
                # 'SETR-PUP',
                # 'SETR-MLA',
                # 'SETR-Naive',
                # 'SegFormer-B0',
                # 'SegFormer-B1',
                # 'SegFormer-B2',
                # 'SegFormer-B3',
                # 'SegFormer-B4',
                # 'SegFormer-B5',
                # 'DLV3+ResNet50',
                # 'DLV3+ResNet101',
                'DLV3+ResNet18',
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

    assert len(set(trg_dataset_list)) == 1 # We compute results for one dset at a time
    trg_dataset_ = trg_dataset_list[0]

    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'

    # Format scenes and conditions
    scenes = ''
    conditions = ''
    for scene, cond in zip(scene_list, cond_list):
        scenes = scenes + scene + ','
        conditions = conditions + cond + ','
    scenes = scenes[:-1]
    conditions = conditions[:-1]


    def get_done_filename(trg_dataset_, model_arch_, results_dir):
        # DONE filename (check if experiment has already been done)
        DONE_name = f'results.pkl'
        model_dir = os.path.join(
        results_dir, 'Cityscapes',
        model_arch_, 'miou_metrics_test_set', trg_dataset_)
        return model_dir, os.path.join(model_dir, DONE_name)

    for model_arch_ in model_arch_list:
        # DONE filename (check if experiment has already been done)
        results_dir, done_filename = get_done_filename(trg_dataset_, model_arch_, root_exp_dir_)
        # Check if done file is present 
        if os.path.isfile(done_filename) and not force_redo_:
            pass
        else:
            print(f'python -u compute_final_metrics.py'+
                      f' --force_redo={force_redo_}' +
                      f' --trg_dataset={trg_dataset_}' +
                      f' --scene={scenes}' +
                      f' --cond={conditions}' +
                      f' --model_arch={model_arch_}' +
                      f' --root_exp_dir={root_exp_dir_}'+
                      f' --results_dir={results_dir}')
