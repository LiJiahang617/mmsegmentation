# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from ..task_modules.prior_generators import MlvlPointGenerator
from .positional_encoding import SinePositionalEncoding
from .transformer import Mask2FormerTransformerEncoder
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dMask


@MODELS.register_module()
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm=None):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = ConvModule(in_chan, in_chan, kernel_size=1, bias=False, norm_cfg=norm)
        self.sigmoid = nn.Sigmoid()
        self.conv = ConvModule(in_chan, out_chan, kernel_size=1, bias=False, norm_cfg=None)
        self.relu = nn.ReLU(inplace=False)

    def init_weights(self) -> None:
        """Initialize weights."""
        caffe2_xavier_init(self.conv_atten, bias=0)
        caffe2_xavier_init(self.conv, bias=0)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.relu(self.conv(x))
        return feat

@MODELS.register_module()
class FeatureAlign(nn.Module):      # With FSM
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm=norm)
        self.offset = ConvModule(out_nc * 2, out_nc, kernel_size=1, bias=False, norm_cfg=norm)
        # remove extra_offset_mask=True for convenient
        self.dcpack_L2 = ModulatedDeformConv2dMask(out_nc, out_nc, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, deformable_groups=8)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self) -> None:
        """Initialize weights."""
        caffe2_xavier_init(self.offset, bias=0)

    def forward(self, feat_l, feat_s, cat=False):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        return feat_align + feat_l


@MODELS.register_module()
class TwinDeformAttnDCNv2PixelDecoder(BaseModule):
    """Twin Pixel decoder with multi-scale deformable attention and DCNv2
    to replace FPN structure.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transformer
            encoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels=None,
                 strides=None,
                 feat_channels: int = 256,
                 out_channels: int = 256,
                 num_outs: int = 3,
                 norm_cfg=None,
                 act_cfg=None,
                 encoder: ConfigType = None,
                 positional_encoding=None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if positional_encoding is None:
            positional_encoding = dict(
                num_feats=128, normalize=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=32)
        if strides is None:
            strides = [4, 8, 16, 32]
        if in_channels is None:
            # in_channels = [256, 512, 1024, 2048]
            in_channels = [512, 1024, 2048, 4096]
        self.strides = strides
        # num_input_levels =4 by default
        self.num_input_levels = len(in_channels)
        # num_encoder_levels = 3 by default
        self.num_encoder_levels = \
            encoder.layer_cfg.self_attn_cfg.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        # encoder is a Deformable DETR self_attn encoder stacked-layer
        self.encoder = Mask2FormerTransformerEncoder(**encoder)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        # fapn-like structure
        self.lateral_convs = ModuleList()
        self.align_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            align_conv = FeatureAlign(feat_channels, feat_channels, norm=norm_cfg)  # ?
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.align_convs.append(align_conv)  # ?
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for m in self.encoder.layers.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, feats: List[Tensor]) -> Tuple[Any, List[Any]]:
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).

        Returns:
            tuple: A tuple containing the following:

                - mask_feature (Tensor): shape (batch_size, c, h, w).
                - multi_scale_features (list[Tensor]): Multi scale \
                        features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        for i in range(self.num_encoder_levels): # 3
            level_idx = self.num_input_levels - i - 1
            # fetch from top to down, low resolution to high resolution
            feat = feats[level_idx]
            # transform channel of feat to 256 (feat_channels)
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding
            # created padding_mask_resized shape -> b, h, w
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            # self.postional_encoding return 'pos' -> [bs, num_feats*2, h, w] -> num_feats -> 128
            pos_embed = self.postional_encoding(padding_mask_resized)
            # self.level_encoding -> [3, 256] or [self.num_encoder_levels, feat_channels]
            level_embed = self.level_encoding.weight[i]
            # level_pos_embed -> [bs, num_feats*2, h, w]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (batch_size, h_i * w_i,  c) -> c -> 256
            feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
            # shape (batch_size, c, h_i, w_i) -> (batch_size, h_i * w_i,  c)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
            # created padding_mask_resized shape -> b, h_i * w_i
            padding_mask_resized = padding_mask_resized.flatten(1)
            # feat_projected -> (batch_size, h_i * w_i,  c)
            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            # level_pos_embed -> (batch_size, h_i * w_i,  c)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            # # reference_points -> (h_i * w_i, 2)
            reference_points_list.append(reference_points)
        # total_num_queries=sum([., h_i * w_i,.])
        # shape (batch_size, total_num_queries),
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape -> (batch_size, total_num_queries, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=1)
        # shape -> (batch_size, total_num_queries, c)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=1)
        device = encoder_inputs.device
        # shape -> (num_encoder_levels, 2), from low resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape -> (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape -> (batch_size, num_total_queries, c)
        memory = self.encoder(
            query=encoder_inputs,
            query_pos=level_positional_encodings,
            key_padding_mask=padding_masks,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_radios)
        # (batch_size, c, num_total_queries)
        memory = memory.permute(0, 2, 1)

        # from low resolution to high resolution
        num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_queries_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]
        # fapn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = self.align_convs[i](cur_feat, outs[-1])
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[:self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features
