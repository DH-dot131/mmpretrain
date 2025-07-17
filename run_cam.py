# run_cam.py
"""
A script to run CAM (Class Activation Mapping) visualization on images using MMPretrain.
This script reads a text file containing image paths and their corresponding labels,
then generates CAM visualizations for each image using the specified model and configuration.
The visualizations are saved to an output directory.
Args:
    --config (str): Path to the model configuration file
    --checkpoint (str): Path to the model checkpoint file
    --out-dir (str): Directory path where CAM visualizations will be saved
    --txt-file (str): Path to text file containing image paths and labels
                      (format: "image_path label" per line)
Example:
    python run_cam.py --config configs/model.py --checkpoint model.pth 
                     --out-dir output/cam --txt-file test.txt
The script uses EigenGradCAM method and targets the 'layer4' of the backbone for visualization.
CAM images are saved with '_cam.jpg' suffix in the specified output directory.
"""
import subprocess
from pathlib import Path
import argparse

# 출력 디렉터리 준비
# out_dir = Path("work_dirs/lstv_classification/fold_1/cam_outputs")
# out_dir.mkdir(parents=True, exist_ok=True)

# # config & checkpoint 경로
# config     = "configs/LSTV_cls/resnet50_LAT_8xb32-fp16_in1k.py"
# checkpoint = "work_dirs/lstv_classification/fold_1/epoch_15.pth"
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Config file path')
parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
parser.add_argument('--out-dir', type=str, required=True, help='Output directory path')
parser.add_argument('--txt-file', type=str, required=True, help='Test txt file path')
args = parser.parse_args()

# Update paths
config = args.config
checkpoint = args.checkpoint
out_dir = Path(args.out_dir)
txt_file = Path(args.txt_file)
out_dir.mkdir(parents=True, exist_ok=True)
# test.txt 파일 읽기
with open(txt_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_path, label = line.split()
        img = Path(img_path)
        save_path = out_dir / f"{img.stem}_cam.jpg"

        cmd = [
            "python", "tools/visualization/vis_cam.py",
            str(img),             # ← 이미지가 맨 앞
            config,
            checkpoint,
            "--method", "EigenGradCAM",
            "--target-layers", "backbone.layer4",  # ← 주석 해제
            "--target-category", label,
            "--aug-smooth",               # ← 하이픈으로 변경
            "--device", "cuda:0",
            "--save-path", str(save_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
