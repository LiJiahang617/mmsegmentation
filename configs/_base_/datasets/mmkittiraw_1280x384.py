# dataset settings
dataset_type = 'MMKittirawDataset'
data_root = '/media/ljh/data/kitti_raw/kitti/training/2011_09_26_drive_0019_sync_02'
sample_scale = (1280, 384)

train_pipeline = [
    dict(type='LoadKittiImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadKittiAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=sample_scale,
         ratio_range=(0.5, 2.0), keep_ratio=True),  # Note: w, h instead of h, w
    dict(type='RandomCrop', crop_size=(384, 1280)),  # h, w
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadKittiImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadKittiAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadKittiImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.jpg',
        ano_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_2',
            depth_path='dense_depth',
            disp_path='disp_2',
            tdisp_path='tdisp', # had an issue in tdisp data, solve it in future
            normal_path='sne',
            seg_map_path='gt_image_2'),
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
        img_suffix='.jpg',
        ano_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_2',
            depth_path='dense_depth',
            disp_path='disp_2',
            tdisp_path='tdisp',  # had an issue in tdisp data, solve it in future
            normal_path='sne',
            seg_map_path='gt_image_2'),
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        img_suffix='.jpg',
        ano_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_2',
            depth_path='dense_depth',
            disp_path='disp_2',
            tdisp_path='tdisp',  # had an issue in tdisp data, solve it in future
            normal_path='sne',
            seg_map_path='gt_image_2'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    output_dir='work_dirs/format_results')
