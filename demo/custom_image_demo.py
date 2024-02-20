# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Optional, Sequence, Union

from mmengine.model import revert_sync_batchnorm
from mmengine.utils import mkdir_or_exist
import mmcv
import numpy as np

from mmseg.models import BaseSegmentor
from mmseg.structures import SegDataSample
from mmseg.apis import inference_model, init_model
from mmseg.visualization import SegLocalVisualizer


def custom_show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       save_dir=None,
                       out_file=None,
                       target_category=None):
    """Customized visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (SegDataSample): The prediction SegDataSample result.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5. Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
        draw_pred (bool): Whether to draw Prediction SegDataSample.
            Defaults to True.
        wait_time (float): The interval of show (s). 0 is the special value
            that means "forever". Defaults to 0.
        show (bool): Whether to display the drawn image.
            Default to True.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        out_file (str, optional): Path to output file. Default to None.

    Returns:
        np.ndarray: the drawn image which channel is RGB.
    """
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(img, str):
        image = mmcv.imread(img, channel_order='rgb')
    else:
        image = img
    if save_dir is not None:
        mkdir_or_exist(save_dir)
    # init visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])
    visualizer.custom_add_datasample(
        name=title,
        image=image,
        data_sample=result,
        draw_gt=draw_gt,
        draw_pred=draw_pred,
        wait_time=wait_time,
        out_file=out_file,
        show=show,
        target_category=target_category)
    vis_img = visualizer.get_image()

    return vis_img


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--save_dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.8,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument(
        '--category', default=[0,1], help='The target categories tha you want to draw.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    custom_show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False,
        save_dir=args.save_dir,
        target_category=args.category)


if __name__ == '__main__':
    main()
