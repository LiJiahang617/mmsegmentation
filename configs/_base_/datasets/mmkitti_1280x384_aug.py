# dataset settings
dataset_type = 'MMKittiDataset'
data_root = '/home/ljh/Desktop/Workspace/mmsegmentation/data/KITTI'
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
        img_suffix='.png',
        modality='normal',
        data_prefix=dict(
            img_path='image_2/training',
            depth_path='lidar_depth_2/training',
            disp_path='disp_2/training',
            tdisp_path='tdisp/training', # had an issue in tdisp data, solve it in future
            normal_path='sne/training',
            seg_map_path='gt_image_2/training'),
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
            img_path='image_2/validation',
            depth_path='lidar_depth_2/validation',
            disp_path='disp_2/validation',
            tdisp_path='tdisp/validation',  # had an issue in tdisp data, solve it in future
            normal_path='sne/validation',
            seg_map_path='gt_image_2/validation'),
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
            img_path='image_2/testing',
            depth_path='lidar_depth_2/testing',
            disp_path='disp_2/testing',
            tdisp_path='tdisp/testing',  # had an issue in tdisp data, solve it in future
            normal_path='sne/testing',
            seg_map_path='gt_image_2/testing'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    output_dir='work_dirs/format_results')
