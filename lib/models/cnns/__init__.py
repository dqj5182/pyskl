# Copyright (c) OpenMMLab. All rights reserved.
from .resnet import ResNet
from .resnet3d_slowonly import ResNet3dSlowOnly
from .x3d import X3D

__all__ = [
    'X3D', 'ResNet', 'ResNet3dSlowOnly'
]