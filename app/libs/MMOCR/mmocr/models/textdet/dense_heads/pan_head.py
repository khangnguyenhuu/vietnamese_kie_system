# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS, build_loss
from mmocr.utils import check_argument
from . import HeadMixin


@HEADS.register_module()
class PANHead(HeadMixin, BaseModule):
    """The class for PANet head."""

    def __init__(
        self,
        in_channels,
        out_channels,
        text_repr_type='poly',  # 'poly' or 'quad'
        downsample_ratio=0.25,
        loss=dict(type='PANLoss'),
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type='Normal', mean=0, std=0.01, override=dict(name='out_conv'))):
        super().__init__(init_cfg=init_cfg)

        assert check_argument.is_type_list(in_channels, int)
        assert isinstance(out_channels, int)
        assert text_repr_type in ['poly', 'quad']
        assert 0 <= downsample_ratio <= 1

        self.loss_module = build_loss(loss)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_repr_type = text_repr_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio
        if loss['type'] == 'PANLoss':
            self.decoding_type = 'pan'
        elif loss['type'] == 'PSELoss':
            self.decoding_type = 'pse'
        else:
            type = loss['type']
            raise NotImplementedError(f'unsupported loss type {type}.')

        self.out_conv = nn.Conv2d(
            in_channels=np.sum(np.array(in_channels)),
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outputs = torch.cat(inputs, dim=1)
        else:
            outputs = inputs
        outputs = self.out_conv(outputs)
        return outputs
