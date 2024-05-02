# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os.path as osp
import time
from mmcv import Config
from mmcv.runner import set_random_seed
from mmcv.utils import get_git_hash

from lib import __version__
from lib.apis import train_model
from lib.datasets import build_dataset
from lib.models import build_model
from lib.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.config = 'assets/joint.py'

    cfg = Config.fromfile(args.config)
    cfg.dist_params = dict(backend='nccl')
    cfg.gpu_ids = range(1)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = 1004
    set_random_seed(seed, deterministic=False)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    model = build_model(cfg.model)

    datasets = [build_dataset(cfg.data.train)]

    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1

    cfg.checkpoint_config.meta = dict(
        pyskl_version=__version__ + get_git_hash(digits=7),
        config=cfg.pretty_text)

    test_option = dict(test_last=True, test_best=True)

    train_model(model, datasets, cfg, validate=True, test=test_option, timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()