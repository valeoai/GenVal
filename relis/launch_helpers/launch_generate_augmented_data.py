'''
Script to launch the command to generate a version of a dataset whose images are 
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)

seed_list=[111]

augmented_dset_dir = 'PATH/TO/SAVE/AUGMENTED/DSET'

def get_done_filename(scene_):
    # DONE filename (check if experiment has already been done)
    DONE_name = f'experiment.DONE'
    file_dir = os.path.join(
    augmented_dset_dir, scene_)
    return os.path.join(file_dir, DONE_name)

epochs = 15
num_transf = 3
counter = 0
for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list[:], scene_list[:], cond_list[:]):
    # DONE filename (check if experiment has already been done)
    done_filename = get_done_filename(scene_)
    # Check if done file is present 
    if os.path.isfile(done_filename) and not force_redo_:
        pass
    else:
        print(f'python -u generate_augmented_data.py' +
                            f' --force_redo={force_redo_}' +
                            f' --trg_dataset={trg_dataset_}' +
                            f' --scene={scene_}' +
                            f' --cond={cond_}' +
                            f' --epochs={epochs}' +
                            f' --num_transf={num_transf}' +
                            f' --augmented_dataset_dir={augmented_dset_dir}{scene_}')
                    

