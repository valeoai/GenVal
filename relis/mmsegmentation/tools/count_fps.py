# Modified version of original script
import argparse
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.apis.inference import init_segmentor, LoadImage


import os
import pickle
import time


mmseg_models_configs = {
    'UpperNetR18': 'mmsegmentation/configs/upernet/upernet_r18_512x1024_80k_cityscapes.py',
    'UpperNetR50': 'mmsegmentation/configs/upernet/upernet_r50_769x769_80k_cityscapes.py',
    'UpperNetR101': 'mmsegmentation/configs/upernet/upernet_r101_769x769_80k_cityscapes.py',
    'SETR-PUP': 'mmsegmentation/configs/setr/setr_vit-large_pup_8x1_768x768_80k_cityscapes.py',
    'SETR-Naive': 'mmsegmentation/configs/setr/setr_vit-large_naive_8x1_768x768_80k_cityscapes.py',
    'SETR-MLA': 'mmsegmentation/configs/setr/setr_vit-large_mla_8x1_768x768_80k_cityscapes.py',
    'SegFormer-B0': 'mmsegmentation/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py',
    'SegFormer-B1': 'mmsegmentation/configs/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes.py',
    'SegFormer-B2': 'mmsegmentation/configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py',
    'SegFormer-B3': 'mmsegmentation/configs/segformer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes.py',
    'SegFormer-B4': 'mmsegmentation/configs/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes.py',
    'SegFormer-B5': 'mmsegmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py',
    'DLV3+ResNet50': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py',
    'DLV3+ResNet101': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py',
    'DLV3+ResNet18': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes.py',
    'Segmenter': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cityscapes.py', # Trained with mmseg
    'ConvNext': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes.py', # Trained with mmseg
    }

mmseg_models_checkpoints = {
    'SETR-PUP': '/gfs-ssd/project/uss/pre_trained_models/SETR/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth',
    'SETR-Naive': '/gfs-ssd/project/uss/pre_trained_models/SETR/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth',
    'SETR-MLA': '/gfs-ssd/project/uss/pre_trained_models/SETR/setr_mla_vit-large_8x1_768x768_80k_cityscapes_20211119_101003-7f8dccbe.pth',
    'SegFormer-B0': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth',
    'SegFormer-B1': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth',
    'SegFormer-B2': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
    'SegFormer-B3': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth',
    'SegFormer-B4': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth',
    'SegFormer-B5': '/gfs-ssd/project/uss/pre_trained_models/SegFormer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
    'DLV3+ResNet50': '/gfs-ssd/project/uss/pre_trained_models/DeepLabV3+/R-50-D8_769x769_80K/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth',
    'DLV3+ResNet101': '/gfs-ssd/project/uss/pre_trained_models/DeepLabV3+/R-101-D8_769x769_80K/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth',
    'DLV3+ResNet18': '/gfs-ssd/project/uss/pre_trained_models/DeepLabV3+/R-18-D8_769x769_80K/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth',
    'Segmenter': '/gfs-ssd/project/uss/pre_trained_models/Segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cityscapes.pth',
    'ConvNext': '/gfs-ssd/project/uss/pre_trained_models/ConvNext/upernet_convnext_large_4x1_fp16_769x769_80k_cityscapes.pth',
     'UpperNetR18': '/gfs-ssd/project/uss/pre_trained_models/UpperNet/R18/upernet_r18_512x1024_80k_cityscapes_20220614_110712-c89a9188.pth',
     'UpperNetR50': '/gfs-ssd/project/uss/pre_trained_models/UpperNet/R50/upernet_r50_769x769_80k_cityscapes_20200607_005107-82ae7d15.pth',
    'UpperNetR101': '/gfs-ssd/project/uss/pre_trained_models/UpperNet/R101/upernet_r101_769x769_80k_cityscapes_20200607_001014-082fc334.pth'
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FPS of segmentors')
    parser.add_argument(
        '--img_path',
        type=str,
        default='/gfs-ssd/project/clara/data/Cityscapes/leftImg8bit/train/aachen/aachen_000100_000019_leftImg8bit.png',
        help='Path to image to be used.')
    parser.add_argument(
        '--results_root',
        type=str,
        default='/gfs-ssd/project/uss/results/Cityscapes/',
        help='File where to save results.')
    args = parser.parse_args()
    return args


def load_model_and_image(args, arch):
    
    ##############################
    ### Load the model
    ##############################
    config = mmseg_models_configs[arch]
    checkpoint = mmseg_models_checkpoints[arch]
    model = init_segmentor(config, checkpoint)
    # Change config of the model to process the dataset
    model.cfg.test_pipeline = [
                                {'type': 'LoadImageFromFile'},
                                {'type': 'MultiScaleFlipAug',
                                         'img_scale': (2048, 1024),
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
    
    
    ##############################
    ### Load the image
    ##############################
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=args.img_path)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
        
    return model, data
        

def main():

    args = parse_args()
    
    arch_list = list(mmseg_models_configs)
    for arch in arch_list:
        # Load image and model
        model, data = load_model_and_image(args, arch)
        # Compute FPS
        with torch.no_grad():
            # First iteration is slower
            result = model(return_loss=False, rescale=True, **data)
            init_time = time.time()
            for i in range(10):
                # Inferecnce
                # forward the model
                result = model(return_loss=False, rescale=True, **data)
            eta = time.time() - init_time

        FPS = 10 / eta
        results_dict = {}
        results_dict['FPS'] = FPS
        results_dict['image_path'] = args.img_path
        
        
        args.results_dir = os.path.join(args.results_root, arch, 'FPS_ablation')
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
