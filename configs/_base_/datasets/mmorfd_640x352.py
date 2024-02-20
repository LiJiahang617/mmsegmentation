# dataset settings
dataset_type = 'MMOrfdDataset'
data_root = '/home/ljh/Desktop/Workspace/depth-to-normal-translator/ORFD_sequence'
sample_scale = (640, 352)

train_pipeline = [
    dict(type='LoadOrfdImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadOrfdAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=sample_scale),  # Note: w, h instead of h, w
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadOrfdImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOrfdAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_data',
            depth_path='dense_depth',
            normal_path='sne',
            seg_map_path='gt_image'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_data',
            depth_path='dense_depth',
            normal_path='sne',
            seg_map_path='gt_image'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_data',
            depth_path='dense_depth',
            normal_path='sne',
            seg_map_path='gt_image'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
