# dataset settings
dataset_type = 'MMCityscapesDataset'
data_root = '/media/ljh/data/cityscapes'
sample_scale = (1024, 512)

train_pipeline = [
    # modality value must be modified, if you choose disp as another modality, you will need to
    # change ano backend to ``tifffile``
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal', anodecode_backend='cv2'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='Resize',
        scale=sample_scale),  # Note: w, h instead of h, w
    dict(type='PackSegInputs')
]
test_pipeline = [
    # modality value must be modified
    dict(type='LoadCityscapesImageFromFile', to_float32=True, modality='normal', anodecode_backend='cv2'),
    dict(type='StackByChannel', keys=('img', 'ano')),
    dict(type='Resize', scale=sample_scale),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
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
        # have to modify next 2 properties at the same time
        modality='normal',
        ano_suffix='_normal.jpg',
        data_prefix=dict(
            img_path='leftImg8bit/train',
            disp_path='left_disp/train',
            normal_path='left_normal/train',
            seg_map_path='gtFine/train'),
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
        # have to modify next 2 properties at the same time
        modality='normal',
        ano_suffix='_normal.jpg',
        data_prefix=dict(
            img_path='leftImg8bit/val',
            disp_path='left_disp/val',
            normal_path='left_normal/val',
            seg_map_path='gtFine/val'),
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
        # have to modify next 2 properties at the same time
        modality='normal',
        ano_suffix='_normal.jpg',
        data_prefix=dict(
            img_path='leftImg8bit/test',
            disp_path='left_disp/test',
            normal_path='left_normal/test',
            seg_map_path='gtFine/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
