# Copyright (c) OpenMMLab. All rights reserved.
from .c3d import C3D
from .resnet import ResNet
from .resnet3d_slowonly import ResNet3dSlowOnly
from .x3d import X3D

__all__ = [
    'C3D', 'X3D', 'ResNet', 'ResNet3dSlowOnly'
]