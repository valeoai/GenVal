# Modified version of original script.
import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

# from ptflops import get_model_complexity_info

from mmseg.models import build_segmentor

import os
import pickle
import torch


mmseg_models_configs = {
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
    'UpperNetR18': 'mmsegmentation/configs/upernet/upernet_r18_512x1024_80k_cityscapes.py',
    'UpperNetR50': 'mmsegmentation/configs/upernet/upernet_r50_769x769_80k_cityscapes.py',
    'UpperNetR101': 'mmsegmentation/configs/upernet/upernet_r101_769x769_80k_cityscapes.py',
    'ConvNext-B-In1K': 'mmsegmentation/configs/convnext/upernet_convnext_base_in1k_fp16_769x769_80k_cityscapes.py',
    'ConvNext-B-In21K': 'mmsegmentation/configs/convnext/upernet_convnext_base_in21k_fp16_769x769_80k_cityscapes.py',
    'BiT-R50x1-In1K': 'mmsegmentation/configs/bit/resnet50x1_in1k.py',
    'BiT-R50x1-In21K': 'mmsegmentation/configs/bit/resnet50x1_in21k.py',
    'SwinLarge': 'mmsegmentation/configs/swin/upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_cityscapes.py',
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('--model_arch', help='architecture to evaluate')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size.')
    parser.add_argument(
        '--results_dir',
        type=str,
        help='File where to save results.')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    config = mmseg_models_configs[args.model_arch]
    cfg = Config.fromfile(config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
        
    flops, params = get_model_complexity_info(model, input_shape)
    
    print(flops)
    print(params)
    
#     from fvcore.nn import FlopCountAnalysis
#     from fvcore.nn import flop_count_table
#     flops = FlopCountAnalysis(model, torch.ones((1, ) + input_shape).cuda())
#     print(flop_count_table(flops))

    results_dict = {}
    results_dict['Input_shape'] = input_shape
    results_dict['Flops'] = flops
    results_dict['Params'] = params
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
