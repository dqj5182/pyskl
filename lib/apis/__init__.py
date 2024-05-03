# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.engine import multi_gpu_test, single_gpu_test

from .inference import inference_recognizer, init_recognizer
from .train import train_model

__all__ = [
    'train_model', 'init_recognizer', 'inference_recognizer', 'multi_gpu_test',
    'single_gpu_test'
]
