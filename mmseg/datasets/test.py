from mmseg.datasets import CityscapesDataset

data_root = '/media/ljh/data/carla_test'
data_prefix=dict(img_path='img_dir/training', seg_map_path='ann_dir/training')
# metainfo 中只保留以下 classes
metainfo=dict(classes=( 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'))
dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, metainfo=metainfo)

print(dataset.metainfo)