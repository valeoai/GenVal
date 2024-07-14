_base_ = [
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# Hack to load timm models from checkpoint
ptcfg = {'file': '/gfs-ssd/project/uss/pre_trained_models/BiT/BiT-M-R50x1-ILSVRC2012.npz'}

# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='TIMMBackbone',
        model_name='resnetv2_50x1_bitm', # resnetv2_50x1_bitm_in21k / resnetv2_101x1_bitm / resnetv2_101x1_bitm_in21k
        pretrained_cfg=ptcfg),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[1, 2, 3, 4], # BiT timm model outputs the stem output as a feature map
                               # (which is not done in ResNet that only has 4 blocks/feature maps)
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=3, # Same here, using 3rd feature map since the first is actually the stem output (not the first block).
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model testing settings
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

# Use 2 images per GPU (with 4 GPU's) like ConvNeXt
data = dict(samples_per_gpu=2) # Note, this needs to be larger than one (or use more than one gpu)
                               # since otherwise BN does not work: you can not have only one sample per batch).
    
# These settings are from ConvNext but ResNet does not have them...
# # fp16 settings
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# # fp16 placeholder
# fp16 = dict()