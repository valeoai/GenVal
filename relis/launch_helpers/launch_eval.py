'''
Helper script to create several launch commands that evaluate a given model on a dataset folder.
Dataset folders are categorized according to 
 - dataset: Cityscapes, IDD or ACDC,
 - scene: corresponding to scene folders in dataset, 
 - condition: descriptive of the condition of the scene can be
        > clean: unmodified images where weather is good (Cityscapes and IDD)
        > fog, rain, snow, night: Conditions in ACDC dataset.
        > augmented: Used for the version of Cityscapes with data augmentations.
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_idd_lists('train_clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_idd_lists('val_clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('test-1-clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_acdc_lists('val', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_acdc_lists('train', trg_dataset_list, scene_list, cond_list)

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

seed_list=[111]


unc_method_list = ['entropy']


def get_done_filename(trg_dataset_, scene_, cond_, unc_method_, model_arch_, root_exp_dir_):
    # DONE filename (check if experiment has already been done)
    trg_sub_folder = f'{trg_dataset_}_{scene_}_{cond_}'
    method_sub_folder = f'uncertainty_{unc_method_}'
    model_arch_sub_folder = model_arch_
    DONE_name = f'experiment.DONE'
    model_dir = os.path.join(
    root_exp_dir_, 'Cityscapes',
    model_arch_sub_folder, trg_sub_folder, method_sub_folder)
    return os.path.join(model_dir, DONE_name)

counter = 0
for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list[:], scene_list[:], cond_list[:]):
    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'
    for seed_ in seed_list:
        for model_arch_ in model_arch_list:
            for unc_method_ in unc_method_list:
                # DONE filename (check if experiment has already been done)
                done_filename = get_done_filename(trg_dataset_, scene_, cond_, unc_method_, model_arch_, root_exp_dir_)
                # Check if done file is present 
                if os.path.isfile(done_filename) and not force_redo_:
                    pass
                else:
                    print(f'python -u eval.py' +
                                f' --force_redo={force_redo_}' +
                                f' --trg_dataset={trg_dataset_}' +
                                f' --scene={scene_}' +
                                f' --cond={cond_}' +
                                f' --uncertainty_method={unc_method_}'+
                                f' --model_arch={model_arch_}' +
                                f' --root_exp_dir={root_exp_dir_}')
                    

