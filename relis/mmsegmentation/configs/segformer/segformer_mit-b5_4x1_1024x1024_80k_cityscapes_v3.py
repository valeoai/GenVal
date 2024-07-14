_base_ = ['../_base_/models/segformer_mit-b0.py',
          '../_base_/datasets/cityscapes_1024x1024.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_80k.py']

path_checkpoint = '/gfs-ssd/project/uss/pre_trained_models/SegFormer/mit_b5_converted.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=path_checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=1)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
