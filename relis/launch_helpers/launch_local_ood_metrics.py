'''
Helper script to create several launch commands that evaluate the OOD detection
(AUROC metric) at the pixel level. Here certain parts of the image corrsponding
to unknown objects can be OOD.
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

for dset in ['IDD']:
    
    trg_dataset_list, scene_list, cond_list = [], [], []

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

    
    IDD_class_names = {
     8: 'autorickshaw',
     16: 'guardrail',
     17: 'billboard',
     23: 'bridge',
    }
    
    ood_classes_list = ['8,16,17,23'] # Which classes to use as OOD (other novel classes will be ignored)

    root_exp_dir_ = 'PATH/TO/EXPERIMENT/FOLDER'

    # Num samples per image to estimate calibration error
    nsamples = 20000
    temperatures = [1]

    trg_dataset_ = dset

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
    
    def get_done_filename(trg_dataset_, model_arch_, root_exp_dir_, nsamples,
                          temp, metric, ood_classes_list):
        if metric == 'entropy':
            metric = '_entropy'
        else:
            metric = ''
            
        ood_classes = '_ood_cls'
        ood_classes_list = ood_classes_list.split(',')
        ood_classes_list = [int(x) for x in ood_classes_list]
        for k in ood_classes_list:
            ood_classes += f'_{IDD_class_names[int(k)]}'
            
        # DONE filename (check if experiment has already been done)
        results_dir = os.path.join(f'{root_exp_dir_}Cityscapes/{model_arch_}/',
                                    f'local_ood_metrics_no_clustering{ood_classes}{metric}/',
                                    f'{trg_dataset_}_samples{nsamples}_T{temp}/')
        DONE_name = f'experiment.DONE'
        return results_dir, os.path.join(results_dir, DONE_name)

    for model_arch_ in model_arch_list:
        for temp in temperatures:
            for metric in ['prob', 'entropy']:
                for ood_classes in ood_classes_list:
                    # DONE filename (check if experiment has already been done)
                    results_dir, done_filename = get_done_filename(trg_dataset_, model_arch_,
                                                                   root_exp_dir_, nsamples, temp, metric, ood_classes)
                    # Check if done file is present 
                    if os.path.isfile(done_filename) and not force_redo_:
                        pass
                    else:
                        print(f'python -u eval_local_ood.py'+
                              f' --force_redo={force_redo_}' +
                              f' --trg_dataset={trg_dataset_}' +
                              f' --trg_dataset_list={trg_datasets}' +
                              f' --scene={scenes}' +
                              f' --cond={conditions}' +
                              f' --confidence_metric={metric}' +
                              f' --model_arch={model_arch_}' +
                              f' --root_exp_dir={root_exp_dir_}'+
                              f' --num_samples={nsamples}'+
                              f' --temp={temp}'+
                              f' --results_dir={results_dir}'+
                              f' --ood_classes={ood_classes}'
                             )
