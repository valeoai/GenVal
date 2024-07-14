_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py', 
]

pretrained_vit = '/tmp-network/project/uss/pre_trained_models/ViT/vit_large_p16_384.pth'
model = dict(
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_vit),
        img_size=(769, 769),
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_heads=16,
        embed_dims=1024,
        num_classes=19,
        align_corners=True,
    ),
    test_cfg=dict(
        mode='slide', crop_size=(769, 769), stride=(513, 513)
    )
)

optimizer = dict(lr=0.001, weight_decay=0.0)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (769, 769)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # Originally used num_gpus: 8 -> batch_size: 8
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)
