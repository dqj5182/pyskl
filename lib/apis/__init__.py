# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.engine import multi_gpu_test, single_gpu_test

from .train import train_model

__all__ = [
    'train_model', 'multi_gpu_test',
    'single_gpu_test'
]
