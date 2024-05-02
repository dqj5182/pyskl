# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import os
import os.path as osp
from mmcv import Config
from mmcv.engine import multi_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from lib.datasets import build_dataloader, build_dataset
from lib.models import build_model
from lib.utils import cache_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    args = parser.parse_args()
    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""

    # build the model and load checkpoint
    model = build_model(cfg.model)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    outputs = multi_gpu_test(model, data_loader, None)

    return outputs


def main():
    args = parse_args()

    args.launcher = 'pytorch'
    args.eval = ['top_k_accuracy', 'mean_class_accuracy']
    args.out = 'result.pkl'

    config_path = 'assets/joint.py'
    cfg = Config.fromfile(config_path)

    out = osp.join(cfg.work_dir, 'result.pkl')

    # Load eval_config from cfg
    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    if not os.path.exists(osp.dirname(out)):
        os.makedirs(osp.dirname(out))
    # mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')

    cfg.data.test.test_mode = True

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    cfg.gpu_ids = range(1)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    outputs = inference_pytorch(args, cfg, data_loader)

    print(f'\nwriting results to {out}')
    dataset.dump_results(outputs, out=out)

    if eval_cfg:
        eval_res = dataset.evaluate(outputs, **eval_cfg)
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
