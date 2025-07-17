# Copyright (c) OpenMMLab. All rights reserved.
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2, os
import torch


import argparse
import os
import os.path as osp
from copy import deepcopy

import mmengine
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--out-item',
        choices=['metrics', 'pred'],
        help='To output whether metrics or predictions. '
        'Defaults to output predictions.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision test')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to enable the Test-Time-Aug (TTA). If the config file '
        'has `tta_pipeline` and `tta_model` fields, use them to determine the '
        'TTA transforms and how to merge the TTA results. Otherwise, use flip '
        'TTA by averaging classification score.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # enable automatic-mixed-precision test
    if args.amp:
        cfg.test_cfg.fp16 = True

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- TTA related args --------------------
    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
        if 'tta_pipeline' not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [test_pipeline[-1]],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
    )

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False

    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    if args.out is None and args.out_item is not None:
        raise ValueError('Please use `--out` argument to specify the '
                         'path of the output file before using `--out-item`.')

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    # === 여기에 추가 ===
    # GPU 사용 여부에 맞춰 use_cuda 설정
    use_cuda = torch.cuda.is_available()
    # 모델의 마지막 conv 계층을 지정 (ResNet 계열 예시)
    target_layers = [runner.model.backbone.layer4[-1]]
    cam = GradCAM(model=runner.model, target_layers=target_layers)

    # CAM 결과를 저장할 폴더
    cam_out_dir = osp.join(cfg.work_dir, 'cam_outputs')
    os.makedirs(cam_out_dir, exist_ok=True)
    # ===================
    

    if args.out and args.out_item in ['pred', None]:
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    metrics = runner.test()
    
    # === CAM 생성 루프 ===
    loader = runner.test_dataloader
    for i, batch in enumerate(loader):
        inputs = batch['inputs'].float().cuda()              # (B, C, H, W)
        data_samples = batch['data_samples']   # List of ClsDataSample      # 메타데이터에 파일명이 들어있다고 가정
        cams = cam(input_tensor=inputs)        # list of (H, W) CAM map

        for j, (img_tensor, cam_map, ds) in enumerate(zip(inputs, cams, data_samples)):
            # raw image array로 복원
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            # CAM 맵 합성
            vis = show_cam_on_image(img_norm, cam_map, use_rgb=True)

            # 저장
            # 메타데이터의 경로 키는 config에 따라 달라질 수 있음
            # 보통 ClassificationDataSample.metainfo['img_path']
            img_path = ds.metainfo.get('img_path', None)
            fname = osp.basename(img_path) if img_path else f'batch{i}_sample{j}.png'
            save_path = osp.join(cam_out_dir, fname.replace('.png','_cam.png'))
            os.makedirs(cam_out_dir, exist_ok=True)
            mmcv.imwrite(vis, save_path)
    # =====================
    
    if args.out and args.out_item == 'metrics':
        mmengine.dump(metrics, args.out)


if __name__ == '__main__':
    main()
