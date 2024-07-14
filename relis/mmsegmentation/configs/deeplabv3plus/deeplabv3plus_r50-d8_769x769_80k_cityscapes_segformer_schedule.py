_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

path_checkpoint = '/gfs-ssd/project/uss/pre_trained_models/DeepLabV3+/R-50-D8_769x769_80K/deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth'

model = dict(
    pretrained=None,
    backbone=dict(init_cfg=dict(
                    type='Pretrained',
                    checkpoint=path_checkpoint,
                    prefix='backbone.')
                 ),
    decode_head=dict(align_corners=True,
#                      init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))
                ),
    auxiliary_head=dict(align_corners=True,
#                         init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg'))
                ),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

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