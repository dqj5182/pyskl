# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import hashlib
import logging
import os
import os.path as osp
import warnings

from mmcv.runner import get_dist_info
from mmcv.utils import get_logger

from ..smp import download_file


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "pyskl".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def cache_checkpoint(filename, cache_dir='.cache'):
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = osp.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not osp.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename

def warning_r0(warn_str):
    rank, _ = get_dist_info()
    if rank == 0:
        warnings.warn(warn_str)
