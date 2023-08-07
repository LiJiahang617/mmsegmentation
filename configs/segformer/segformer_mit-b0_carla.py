_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/carla_1280x704.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(704, 1280))

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/home/ljh/Desktop/Workspace/mmsegmentation/pretrain/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth'))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
