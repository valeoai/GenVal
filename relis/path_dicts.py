### Here we define paths to datasets, model configs and checkpoints

# CITYSCAPES_ROOT = 'mmsegmentation/data/cityscapes'
CITYSCAPES_ROOT = '../datasets/cityscapes'
ACDC_ROOT = '../datasets/ACDC'
IDD_ROOT = '../datasets/IDD_Segmentation'
CITYSCAPES_FOG = '../images/style_transfer/fog'
CITYSCAPES_NIGHT = '../images/style_transfer/night'
CITYSCAPES_RAIN = '../images/style_transfer/rain'
CITYSCAPES_SNOW = '../images/style_transfer/snow'
CITYSCAPES_INDIA = '../images/style_transfer/india'


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
    'SegFormer-B5_v2': 'mmsegmentation/configs/segformer/segformer_mit-b5_4x1_1024x1024_80k_cityscapes_v2.py',
    'SegFormer-B5_v3': 'mmsegmentation/configs/segformer/segformer_mit-b5_4x1_1024x1024_80k_cityscapes_v3.py',
    'DLV3+ResNet18': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes.py',
    'DLV3+ResNet50': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py',
    'DLV3+ResNet101': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py',
    'Segmenter': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cityscapes.py',
    'ConvNext': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes.py',
    'UpperNetR18': 'mmsegmentation/configs/upernet/upernet_r18_512x1024_80k_cityscapes.py',
    'UpperNetR50': 'mmsegmentation/configs/upernet/upernet_r50_769x769_80k_cityscapes.py',
    'UpperNetR101': 'mmsegmentation/configs/upernet/upernet_r101_769x769_80k_cityscapes.py',
    'ConvNext-B-In1K': 'mmsegmentation/configs/convnext/upernet_convnext_base_in1k_fp16_769x769_80k_cityscapes.py',
    'ConvNext-B-In21K': 'mmsegmentation/configs/convnext/upernet_convnext_base_in21k_fp16_769x769_80k_cityscapes.py',
    'SwinLarge': 'mmsegmentation/configs/swin/upernet_swin_large_patch4_window7_512x512_pretrain_224x224_22K_160k_cityscapes.py',
    'ANN-R50': 'mmsegmentation/configs/ann/ann_r50-d8_769x769_80k_cityscapes.py',
    'ANN-R101': 'mmsegmentation/configs/ann/ann_r101-d8_769x769_80k_cityscapes.py',
    'APCNet-R50': 'mmsegmentation/configs/apcnet/apcnet_r50-d8_769x769_80k_cityscapes.py',
    'APCNet-R101': 'mmsegmentation/configs/apcnet/apcnet_r101-d8_769x769_80k_cityscapes.py',
    'BiSeNetV1-R18': 'mmsegmentation/configs/bisenetv1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py',
    'BiSeNetV1-R50': 'mmsegmentation/configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py',
    'BiSeNetV2-FCN': 'mmsegmentation/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py',
    'CCNet-R50': 'mmsegmentation/configs/ccnet/ccnet_r50-d8_769x769_80k_cityscapes.py',
    'CCNet-R101': 'mmsegmentation/configs/ccnet/ccnet_r101-d8_769x769_80k_cityscapes.py',
    'GCNet-R50': 'mmsegmentation/configs/gcnet/gcnet_r50-d8_769x769_80k_cityscapes.py',
    'GCNet-R101': 'mmsegmentation/configs/gcnet/gcnet_r101-d8_769x769_80k_cityscapes.py',
    'ICNet-R18': 'mmsegmentation/configs/icnet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes.py',
    'ICNet-R50': 'mmsegmentation/configs/icnet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes.py',
    'ICNet-R101': 'mmsegmentation/configs/icnet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes.py',
    'MobileNetV3': 'mmsegmentation/configs/mobilenet_v3/lraspp_m-v3-d8_512x1024_320k_cityscapes.py',
    'PSPNet-R18': 'mmsegmentation/configs/pspnet/pspnet_r18-d8_769x769_80k_cityscapes.py',
    'PSPNet-R50': 'mmsegmentation/configs/pspnet/pspnet_r50-d8_769x769_80k_cityscapes.py',
    'PSPNet-R101': 'mmsegmentation/configs/pspnet/pspnet_r101-d8_769x769_80k_cityscapes.py',
    'SemFPN-R50': 'mmsegmentation/configs/sem_fpn/fpn_r50_512x1024_80k_cityscapes.py',
    'SemFPN-R101': 'mmsegmentation/configs/sem_fpn/fpn_r101_512x1024_80k_cityscapes.py'
    }


# Paths to model checkpoints: Models should be downloaded from mmseg or locally trained
mmseg_models_checkpoints = {
    'SETR-PUP': '../checkpoints/SETR/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth',
    'SETR-Naive': '../checkpoints/SETR/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth',
    'SETR-MLA': '../checkpoints/SETR/setr_mla_vit-large_8x1_768x768_80k_cityscapes_20211119_101003-7f8dccbe.pth',
    'SegFormer-B0': '../checkpoints/SegFormer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth',
    'SegFormer-B1': '../checkpoints/SegFormer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth',
    'SegFormer-B2': '../checkpoints/SegFormer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
    'SegFormer-B3': '../checkpoints/SegFormer/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth',
    'SegFormer-B4': '../checkpoints/SegFormer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth',
    'SegFormer-B5': '../checkpoints/SegFormer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
    'SegFormer-B5_v2': '../checkpoints/SegFormer/SegFormer-B5_v2_ckpt.pth',
    'SegFormer-B5_v3': '../checkpoints/SegFormer/SegFormer-B5_v3_ckpt.pth',
    'DLV3+ResNet18': '../checkpoints/DLV3+/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth',
    'DLV3+ResNet50': '../checkpoints/DLV3+/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth',
    'DLV3+ResNet101': '../checkpoints/DLV3+/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth',
    'Segmenter': '../checkpoints/Segmenter/Segmenter_ckpt.pth',
    'ConvNext': '../checkpoints/ConvNext/ConvNext_ckpt.pth',
    'UpperNetR18': '../checkpoints/UPerNet/UpperNetR18_ckpt.pth',
    'UpperNetR50': '../checkpoints/UPerNet/upernet_r50_769x769_80k_cityscapes_20200607_005107-82ae7d15.pth',
    'UpperNetR101': '../checkpoints/UPerNet/upernet_r101_769x769_80k_cityscapes_20200607_001014-082fc334.pth',
    'ConvNext-B-In1K': '../checkpoints/ConvNext/ConvNext-B-In1K_ckpt.pth',
    'ConvNext-B-In21K': '../checkpoints/ConvNext/ConvNext-B-In21K_ckpt.pth',
    'SwinLarge': '../checkpoints/Swin/SwinLarge_ckpt.pth',
    'ANN-R50': '../checkpoints/ANN/ann_r50-d8_769x769_80k_cityscapes_20200607_044426-cc7ff323.pth',
    'ANN-R101': '../checkpoints/ANN/ann_r101-d8_769x769_80k_cityscapes_20200607_013713-a9d4be8d.pth',
    'APCNet-R50': '../checkpoints/APCNet/apcnet_r50-d8_769x769_80k_cityscapes_20201214_115718-7ea9fa12.pth',
    'APCNet-R101': '../checkpoints/APCNet/apcnet_r101-d8_769x769_80k_cityscapes_20201214_115716-a7fbc2ab.pth',
    'BiSeNetV1-R18': '../checkpoints/BiSeNetV1/bisenetv1_r18-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210905_220251-8ba80eff.pth',
    'BiSeNetV1-R50': '../checkpoints/BiSeNetV1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth',
    'BiSeNetV2-FCN': '../checkpoints/BiSeNetV2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth',
    'CCNet-R50': '../checkpoints/CCNet/ccnet_r50-d8_769x769_80k_cityscapes_20200617_010421-73eed8ca.pth',
    'CCNet-R101': '../checkpoints/CCNet/ccnet_r101-d8_769x769_80k_cityscapes_20200618_011502-ad3cd481.pth',
    'GCNet-R50': '../checkpoints/GCNet/gcnet_r50-d8_769x769_80k_cityscapes_20200619_092516-4839565b.pth',
    'GCNet-R101': '../checkpoints/GCNet/gcnet_r101-d8_769x769_80k_cityscapes_20200619_092628-8e043423.pth',
    'ICNet-R18': '../checkpoints/ICNet/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes_20210926_052702-619c8ae1.pth',
    'ICNet-R50': '../checkpoints/ICNet/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715-ce310aea.pth',
    'ICNet-R101': '../checkpoints/ICNet/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612-9484ae8a.pth',
    'MobileNetV3': '../checkpoints/MobileNetV3/lraspp_m-v3-d8_512x1024_320k_cityscapes_20201224_220337-cfe8fb07.pth',
    'PSPNet-R18': '../checkpoints/PSPNet/pspnet_r18-d8_769x769_80k_cityscapes_20201225_021458-3deefc62.pth',
    'PSPNet-R50': '../checkpoints/PSPNet/pspnet_r50-d8_769x769_80k_cityscapes_20200606_210121-5ccf03dd.pth',
    'PSPNet-R101': '../checkpoints/PSPNet/pspnet_r101-d8_769x769_80k_cityscapes_20200606_225055-dba412fa.pth',
    'SemFPN-R50': '../checkpoints/SemFPN/fpn_r50_512x1024_80k_cityscapes_20200717_021437-94018a0d.pth',
    'SemFPN-R101': '../checkpoints/SemFPN/fpn_r101_512x1024_80k_cityscapes_20200717_012416-c5800d4c.pth'
    }
