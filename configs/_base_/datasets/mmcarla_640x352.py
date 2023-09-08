# dataset settings
dataset_type = 'MMCarlaDataset'
data_root = '/media/ljh/data/carla_backup/carla_v2/2023_05_16/Tiled_V2/WetCloudyNight/roll0/base05/'
train_pipeline = [
    dict(type='LoadMultimodalImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadCarlaAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=(640, 352)),  # Note: w, h instead of h, w
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultimodalImageFromFile', to_float32=True, modality='normal'),  # modality value must be modified
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=(640, 352)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadCarlaAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
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
            img_path='rgb_front_left',
            depth_path='depth_left',
            disp_path='disparity_left',
            tdisp_path='tdisp', # had an issue in tdisp data, solve it in future
            normal_path='normal_left',
            seg_map_path='semantic_segmentation_left'),
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
            img_path='rgb_front_left',
            depth_path='depth_left',
            disp_path='disparity_left',
            tdisp_path='tdisp',  # had an issue in tdisp data, solve it in future
            normal_path='normal_left',
            seg_map_path='semantic_segmentation_left'),
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
            img_path='rgb_front_left',
            depth_path='depth_left',
            disp_path='disparity_left',
            tdisp_path='tdisp',  # had an issue in tdisp data, solve it in future
            normal_path='normal_left',
            seg_map_path='semantic_segmentation_left'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    format_only=True,
    output_dir='work_dirs/carla_demo')