# dataset settings
dataset_type = 'CarlaDataset'
data_root = '/media/ljh/data/carla_test'
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadCarlaAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=(1280, 704)),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=(1280, 704)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        data_prefix=dict(
            img_path='img_dir/training',
            seg_map_path='ann_dir/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        data_prefix=dict(
            img_path='img_dir/validation',
            seg_map_path='ann_dir/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
