_base_ = ['../_base_/models/segformer_mit-b0.py',
          '../_base_/datasets/cityscapes_1024x1024.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_80k.py']

path_checkpoint = '/gfs-ssd/project/uss/pre_trained_models/SegFormer/mit_b5.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=path_checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=1)