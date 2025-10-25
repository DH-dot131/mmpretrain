#!/usr/bin/env python3
"""
Batch Test Script for MMDetection/MMPretrain

This script automates running `tools/test.py` on all checkpoints in a given directory.
For each `epoch_{n}.pth` file in the checkpoint directory, it:
 1. Overrides the `DumpResults` evaluator's output path to `results-ep{n}.pkl`.
 2. Calls `tools/test.py` with the provided config and checkpoint.
 3. Saves all outputs under a `results/` subfolder (or a custom one).

Usage:
  python tools/batch_test.py \
      configs/your_config.py \
      work_dirs/your_experiment_dir \
      [--results-dir work_dirs/your_experiment_dir/results]

"""
import os
import glob
import subprocess
import argparse
import sys

# Windows 환경에서 인코딩 문제 해결을 위한 환경 변수 설정
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    # mmengine의 collect_env에서 발생하는 인코딩 문제 해결
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch-run tools/test.py over all epoch_*.pth checkpoints'
    )
    parser.add_argument(
        'config', help='Path to the config file (e.g., configs/your_config.py)'
    )
    parser.add_argument(
        'checkpoint_dir', help='Directory containing epoch_*.pth files'
    )
    parser.add_argument(
        '--results-dir',
        help='Directory where results will be saved. Defaults to <checkpoint_dir>/results',
        default=None
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    ckpt_dir = args.checkpoint_dir
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' does not exist.")
        sys.exit(1)
    
    # Check if checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoint directory '{ckpt_dir}' does not exist.")
        sys.exit(1)
    
    # Check if tools/test.py exists
    test_script = 'tools/test.py'
    if not os.path.exists(test_script):
        print(f"Error: Test script '{test_script}' does not exist.")
        sys.exit(1)
    
    # default results folder inside checkpoint dir
    results_dir = args.results_dir or os.path.join(ckpt_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # find all epoch_*.pth
    pattern = os.path.join(ckpt_dir, '*.pth')
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        print(f"No checkpoints found matching {pattern}")
        return

    print(f"Found {len(ckpts)} checkpoints to test")
    
    for i, ckpt in enumerate(ckpts, 1):
        epoch = os.path.basename(ckpt)  # 파일명에서 확장자 제거
        out_pkl = os.path.join(results_dir, f'results-{epoch}.pkl')
        vis_dir = os.path.join(results_dir, 'vis_test', f'ep-{epoch}')
        # override DumpResults evaluator's output path
        cfg_opt = [f"test_evaluator.3.out_file_path={out_pkl}", 
                #    f"default_hooks.visualization.out_dir={vis_dir}"
                   ]

        cmd = [
            'python', test_script,
            config_path,
            ckpt,
            '--cfg-options', *cfg_opt
        ]
        print(f"\n==> Testing epoch {epoch} ({i}/{len(ckpts)}): {ckpt}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Windows 환경에서 인코딩 문제를 해결하기 위해 바이너리 모드 사용
            env = os.environ.copy()
            if os.name == 'nt':  # Windows
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
                env['PYTHONUTF8'] = '1'
                # 추가 환경 변수 설정
                env['PYTHONHASHSEED'] = '0'
                env['PYTHONUNBUFFERED'] = '1'
            
            # shell=True를 사용하여 더 안정적인 실행
            cmd_str = ' '.join(cmd)
            result = subprocess.run(cmd_str, check=True, capture_output=True, env=env, shell=True)
            print(f"--> Successfully saved results to {out_pkl}")
        except subprocess.CalledProcessError as e:
            print(f"--> Error testing epoch {epoch}:")
            print(f"Return code: {e.returncode}")
            try:
                error_output = e.stderr.decode('utf-8', errors='ignore')
            except:
                error_output = str(e.stderr)
            print(f"Error output: {error_output}")
            continue
        except FileNotFoundError:
            print(f"--> Error: Could not find 'python' executable")
            sys.exit(1)
        
if __name__ == '__main__':
    main()
