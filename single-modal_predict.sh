# python demo/image_demo.py \
# /remote-home/jhli/TIV/TIV/data/carla_v3/images/testing/26_599102983.png \
# /remote-home/jhli/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_carla-352x640.py \
# /remote-home/jhli/ICRA_2024_used/experiments_record/work_dirs/hrnet_fcn_hr48_carla-352x640/best_mIoU_epoch_34.pth \
# --save_dir single_carla_demo/hrnet_night
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/frankfurt_000000_000294_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
--save_dir val_cityscapes_demo2/hrnet_frankfurt_000000_000294
#python demo/custom_image_demo.py \
#/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/lindau_000046_000019_leftImg8bit.png \
#/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
#/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
#--save_dir val_cityscapes_demo/hrnet_lindau_000046_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000000_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
--save_dir val_cityscapes_demo2/hrnet_munster_000000_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000114_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
--save_dir val_cityscapes_demo2/hrnet_munster_000114_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000162_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
--save_dir val_cityscapes_demo2/hrnet_munster_000162_000019
# =================================================================================================================
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/frankfurt_000000_000294_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/deeplabv3plus_r101_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/deeplabv3plus_r101_cityscapes-512x1024/best_mIoU_epoch_33.pth \
--save_dir val_cityscapes_demo2/deeplabv3plus_frankfurt_000000_000294
#python demo/custom_image_demo.py \
#/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/lindau_000046_000019_leftImg8bit.png \
#/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
#/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
#--save_dir val_cityscapes_demo/hrnet_lindau_000046_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000000_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/deeplabv3plus_r101_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/deeplabv3plus_r101_cityscapes-512x1024/best_mIoU_epoch_33.pth \
--save_dir val_cityscapes_demo2/deeplabv3plus_munster_000000_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000114_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/deeplabv3plus_r101_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/deeplabv3plus_r101_cityscapes-512x1024/best_mIoU_epoch_33.pth \
--save_dir val_cityscapes_demo2/deeplabv3plus_munster_000114_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000162_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/deeplabv3plus_r101_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/deeplabv3plus_r101_cityscapes-512x1024/best_mIoU_epoch_33.pth \
--save_dir val_cityscapes_demo2/deeplabv3plus_munster_000162_000019
# =================================================================================================================
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/frankfurt_000000_000294_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/mask2former_swin-l_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/mask2former_swin-l_cityscapes-512x1024/epoch_20.pth \
--save_dir val_cityscapes_demo2/mask2former_frankfurt_000000_000294
#python demo/custom_image_demo.py \
#/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/lindau_000046_000019_leftImg8bit.png \
#/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/hrnet_fcn_hr48_cityscapes-512x1024.py \
#/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/hrnet_fcn_hr48_cityscapes-512x1024/best_mIoU_epoch_31.pth \
#--save_dir val_cityscapes_demo/hrnet_lindau_000046_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000000_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/mask2former_swin-l_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/mask2former_swin-l_cityscapes-512x1024/epoch_20.pth \
--save_dir val_cityscapes_demo2/mask2former_munster_000000_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000114_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/mask2former_swin-l_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/mask2former_swin-l_cityscapes-512x1024/epoch_20.pth \
--save_dir val_cityscapes_demo2/mask2former_munster_000114_000019
python demo/custom_image_demo.py \
/media/ljh/Kobe24/samsung_touch7/Cityscapes/images/val/munster_000162_000019_leftImg8bit.png \
/home/ljh/Desktop/Workspace/mmsegmentation/configs/roadformer_single-modal_sup/mask2former_swin-l_cityscapes-512x1024.py \
/home/ljh/Desktop/Workspace/mmsegmentation/work_dirs/mask2former_swin-l_cityscapes-512x1024/epoch_20.pth \
--save_dir val_cityscapes_demo2/mask2former_munster_000162_000019
