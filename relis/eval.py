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
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD

from metrics_helpers import compute_mIoU_single_image, compute_acc_single_image, \
compute_mIoU_fromlist, compute_acc_fromlist

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

        assert len(self.args.cond.split('-')) == len(self.args.scene.split('-'))

        self.image_ops = ImageOps()
        self.uncertainty_ops = UncertaintyOps()

        w_trg, h_trg = map(int, self.args.input_size.split(','))
        self.input_size = (w_trg, h_trg)


    def eval(self):

        """
        Method to eval a model sample by sample on a given
        sequence, and compute each sample's uncertainty.
        All parameters setup by the user (args).
        """

        cudnn.enabled = True
        gpu = self.args.gpu
        #torch.use_deterministic_algorithms(True) # TODO fix weird error
        cudnn.benchmark = True

        summary_dict = {
                        'pixel_acc_images':[],
                        'mean_class_acc_images':[],
                        'miou_images':[],
                        'present_class_miou_images':[],
                        'pixel_acc_final':None,
                        'mean_class_acc_final':None,
                        'miou_final':None
                        }

        self.args.num_steps = len(self.trg_eval_loader)

        if self.args.batch_size != 1:
            raise NotImplementedError("Code only supported for BS = 1 for the moment")        
        for i_iter, trg_batch in enumerate(self.trg_eval_loader):

            # Collect one batch (single image if bs=1)
            trg_image, trg_labels, _, trg_image_name = trg_batch
            trg_image = np.array(trg_image.squeeze()) # Convert image from torch to np array.
            trg_image_name = trg_image_name[0]

            # NOTE horrible hack to check if no label is available
            if -999 in trg_labels:
                trg_labels = None

            if 'Mask' in self.args.model_arch: # Use Detectron2 library
                from detectron2.data.detection_utils import read_image
                image = read_image(trg_image_name, format="BGR")
                trg_logits = self.model(image)['sem_seg']
                trg_logits_cpu = trg_logits.cpu().numpy()
                
            else: # Use mmsegmentation model
                from mmsegmentation.mmseg.apis import inference_segmentor
                trg_logits = inference_segmentor(self.model, trg_image_name, output_logits=True)
                trg_logits_cpu = trg_logits.cpu().data[0].numpy() #1 torch.Size([C, H, W])

            # ------- computing initial miou ------------
            

            # process the image for further operations (mious, acc, saving it etc.)
            trg_logits_cpu = trg_logits_cpu.transpose(1,2,0) #2 (H, W, C)
            trg_pred = np.asarray(np.argmax(trg_logits_cpu, axis=2), dtype=np.uint8) #3 (H, W, C)


            ########### SEM. SEGM. METRICS #################
            if trg_labels is not None:
                trg_labels = np.squeeze(trg_labels.detach().cpu().numpy()).astype(np.uint8)
                mious = compute_mIoU_single_image(trg_labels, trg_pred, self.args)
                pixel_acc, mean_class_acc = compute_acc_single_image(trg_labels, trg_pred)
                present_class_mious = np.copy(mious)
                image_gt = np.unique(trg_labels)
                image_gt = image_gt[image_gt!=255]
                present_class_mious[np.delete(np.arange(self.args.num_classes), image_gt)] = np.nan
            else:
                mious = None
                pixel_acc = None
                mean_class_acc = None
                present_class_mious = None

            # --- appending stuff to dict -----------
            #if trg_labels is not None:
            summary_dict['pixel_acc_images'].append(pixel_acc)
            summary_dict['mean_class_acc_images'].append(mean_class_acc)
            summary_dict['miou_images'].append(mious)
            summary_dict['present_class_miou_images'].append(present_class_mious)
            # --------------------------------------

            ########### UNCERTAINTY ########################
            # compute current sample's uncertainty
            if self.args.uncertainty_method == 'entropy':
                if len(trg_logits.shape) == 3: # Detectron model does not return batch dim for single images
                    trg_logits = trg_logits.unsqueeze(0)
                uncertainty_map = self.uncertainty_ops.compute_output_entropy(
                        trg_logits, False, False)
            else:
                raise ValueError("f'Unknown mode {self.args.uncertainty_method}'")

            # processing uncertainty map
            uncertainty_map = uncertainty_map.detach().cpu().numpy().squeeze()
            uncertainty_image = 255. * uncertainty_map / np.log(self.args.num_classes) # TODO set this right
            uncertainty_image = uncertainty_image.astype(np.uint8)
            
            ################################################

            if (trg_labels is not None): 
                mious_print = np.nanmean(mious)
                present_class_mious_print = np.nanmean(present_class_mious)
                pixel_acc_print = pixel_acc
                mean_class_acc_print = mean_class_acc
            else:
                mious_print = -999.
                present_class_mious_print = -999.
                pixel_acc_print = -999.
                mean_class_acc_print = -999.

            if i_iter%1==0:
                # --- PRINTING RESULTS ------
                print(f'iter = {i_iter:4d}/{self.args.num_steps:5d}, '+ \
                        f'mious = {mious_print:.3f}, '+ \
                        f'present_class_mious = {present_class_mious_print:.3f}, ' + \
                        f'pixel_acc = {pixel_acc_print:.3f}, ' + \
                        f'mean_class_acc = {mean_class_acc_print:.3f}'
                        )

            ########### SAVING IMAGES ########################
            
            trg_image_rgb = self.image_ops.process_image_for_saving(trg_image)
            trg_pred_rgb = self.image_ops.colorize_mask(trg_pred)

            if trg_labels is not None:
                # colorizing labels
                trg_labels_rgb = self.image_ops.colorize_mask(trg_labels)
            else:
                # quick solution replicating the prediction in case no GT available
                trg_labels_rgb = trg_image

            # self.image_ops.save_concat_image(
            #     trg_image_rgb, trg_labels_rgb, trg_pred_rgb, uncertainty_image,
            #     self.output_images_dir, f'{i_iter:07d}_concat.png')

            # image_name = trg_image_name.split('/')[-1]
            # Saving labels
            Image.fromarray(trg_pred).save(os.path.join(self.output_images_dir, f'{i_iter:07d}_label.png'))
            # Saving rgb maps
            # trg_pred_rgb.save(os.path.join(self.output_images_dir, f'{i_iter:07d}_color.png'))
            # Saving uncertainty maps
            # uncertainty_image = Image.fromarray(uncertainty_image)
            # uncertainty_image.save(os.path.join(self.output_images_dir, f'{i_iter:07d}_uncertainty_map.png'))
            # with open(os.path.join(self.output_images_dir, f'{i_iter:07d}_uncertainty_values.pkl'),'wb') as f:
            #     pickle.dump(uncertainty_map, f, pickle.HIGHEST_PROTOCOL)
            ##################################################

        print('End of evaluation.')

        with open(os.path.join(self.model_dir, self.summary_name),'wb') as f:
            pickle.dump(summary_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.model_dir, self.DONE_name),'wb') as f:
            print('Saving end of training file')


   
    def build_model(self):
        # Create network
        config = mmseg_models_configs[self.args.model_arch]
        checkpoint = mmseg_models_checkpoints[self.args.model_arch]
        if 'Mask' in self.args.model_arch: # Use Detectron2 library
            
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config

            cfg = get_cfg()
            add_deeplab_config(cfg)
            if 'Mask2' in self.args.model_arch:
                from Mask2Former.mask2former import add_maskformer2_config
                add_maskformer2_config(cfg)
            else:
                from MaskFormer.mask_former import add_mask_former_config
                add_mask_former_config(cfg)
            cfg.merge_from_file(config)
#             cfg.merge_from_list(args.opts)
            cfg.freeze()
            self.model = DefaultPredictor(cfg) # To be used on a single GPU
            
        else: # Use MMSegmentation library
            
            from mmsegmentation.mmseg.apis import init_segmentor, extract_backbone_features
            # Create network
            config = mmseg_models_configs[self.args.model_arch]
            checkpoint = mmseg_models_checkpoints[self.args.model_arch]
            self.model = init_segmentor(config, checkpoint,
                                        device=f'cuda:{self.args.gpu}')
            # Set model decoder to provide features
            self.model.decode_head.provide_features = True

            # Set up config of the model to process the dataset
            self.model.cfg.test_pipeline = [
                                            {'type': 'LoadImageFromFile'},
                                            {'type': 'MultiScaleFlipAug',
                                                'img_scale': (self.input_size[0], self.input_size[1]),
                                                'flip': False,
                                                'transforms': [
                                                    {'type': 'Resize', 'keep_ratio': True},
                                                    {'type': 'RandomFlip'},
                                                    {'type': 'Normalize',
                                                        'mean': [123.675, 116.28, 103.53], # TODO: Should we adapt it to target dsets?
                                                        'std': [58.395, 57.12, 57.375],
                                                        'to_rgb': True},
                                                    {'type': 'ImageToTensor', 'keys': ['img']},
                                                    {'type': 'Collect', 'keys': ['img']}
                                                ]
                                            }
                                        ]
        print('Done')
            

    def loss_calc(self, pred, label, gpu):

        """
        This function returns cross entropy loss for semantic segmentation
        """

        # out shape batch_size x channels x h x w -> batch_size x channels x h x w
        # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
        label = Variable(label.long()).cuda(gpu)
        criterion = CrossEntropy2d().cuda(gpu)

        return criterion(pred, label)


    def setup_experiment_folder(self):

        """
        Method to define model folder's name and create it, and to
        define the name of the output files created at end of training.
        """

        trg_sub_folder = f'{self.args.trg_dataset}_{self.args.scene}_{self.args.cond}'
        method_sub_folder = f'uncertainty_{self.args.uncertainty_method}'
        model_arch_sub_folder = self.args.model_arch

        self.DONE_name = f'experiment.DONE'
        self.summary_name = f'summary.pkl'

        self.model_dir = os.path.join(
                self.args.root_exp_dir, self.args.src_dataset,
                model_arch_sub_folder, trg_sub_folder, method_sub_folder)

        # check if experiment/testing was done already -------------------------------
        if os.path.isfile(
                    os.path.join(self.model_dir, self.DONE_name)) \
                and not self.args.force_redo:
            print('DONE file present -- evaluation has already been carried out')
            print(os.path.join(self.model_dir, self.DONE_name))
            exit(0)
        # -----------------------------------------------------------------------------

        self.output_images_dir = os.path.join(
                self.model_dir, 'output_images')

        print(f'EXP ---> {self.model_dir}')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.output_images_dir):
            os.makedirs(self.output_images_dir)


    def setup_target_data_loader(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        # (can also be a single environment)
        scene_list = [self.args.scene]
        cond_list = [self.args.cond]
            

        if self.args.trg_dataset=='Cityscapes':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_ROOT,
                    scene_list, cond_list)
        
        elif self.args.trg_dataset=='Cityscapes_fog':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_FOG,
                    scene_list, cond_list)
        
        elif self.args.trg_dataset=='Cityscapes_night':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_NIGHT,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_rain':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_RAIN,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_snow':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_SNOW,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='Cityscapes_india':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_INDIA,
                    scene_list, cond_list)
            
        elif self.args.trg_dataset=='Cityscapes_syn':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_SYN,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='ACDC':
            self.trg_parent_set = ACDC(
                    ACDC_ROOT, scene_list, cond_list,
                    batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='IDD':
            self.trg_parent_set = IDD(
                    IDD_ROOT, scene_list, batch_size=self.args.batch_size)

        else:
            raise ValueError(f'Unknown dataset {self.args.dataset}')

        self.trg_eval_loader = data.DataLoader(
                self.trg_parent_set, batch_size=self.args.batch_size,
                shuffle=False, pin_memory=True)


if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser()

    # what uncertainty to compute
    parser.add_argument("--uncertainty_method", type=str, default='entropy',
                        help="available options for uncertainty : entropy")

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
    parser.add_argument("--root_exp_dir", type=str, default='results/debug/',
                        help="Where to save predictions.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")

    # for target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
    parser.add_argument("--scene", type=str, default='aachen',
                        help="Scene, depends on specific datasets.")
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
    elif args.trg_dataset.endswith(("fog", "night", "rain", "snow", "india", "syn")):
        args.input_size = '1024,1024'
    else:
        raise NotImplementedError("Input size unknown")

    npr.seed(args.seed)

    solver_ops = SolverOps(args)

    print('Setting up experiment folder')
    solver_ops.setup_experiment_folder()

    print('Setting up data target loader')
    solver_ops.setup_target_data_loader()

    print(f'Building {args.model_arch} model')
    solver_ops.build_model()

    print('Start evaluating')
    solver_ops.eval()

