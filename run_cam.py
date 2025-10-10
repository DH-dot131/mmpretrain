# run_cam.py
"""
A script to run CAM (Class Activation Mapping) visualization on images using MMPretrain.
This script reads a text file containing image paths and their corresponding labels,
then generates CAM visualizations for each image using the specified model and configuration.
The visualizations are saved to an output directory.

This script uses batch processing mode of vis_cam.py to efficiently process multiple images
with automatic GPU memory management.

Args:
    --config (str): Path to the model configuration file
    --checkpoint (str): Path to the model checkpoint file
    --out-dir (str): Directory path where CAM visualizations will be saved
    --txt-file (str): Path to text file containing image paths and labels
                      (format: "image_path label" per line)
    --method (str): CAM method to use (default: EigenCAM)
    --target-layers (str): Target layer for CAM (default: backbone.layer4)
    --device (str): Device to use (default: cuda)
    --clear-cache-interval (int): Clear GPU cache every N images (default: 10)

Example:
    python run_cam.py --config configs/model.py --checkpoint model.pth 
                     --out-dir output/cam --txt-file test.txt
                     --method EigenCAM --device cuda

CAM images are saved with '_cam.jpg' suffix in the specified output directory.
"""
import subprocess
from pathlib import Path
import argparse

# Command line arguments
parser = argparse.ArgumentParser(description='Batch CAM visualization using MMPretrain')
parser.add_argument('--config', type=str, required=True, help='Config file path')
parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
parser.add_argument('--out-dir', type=str, required=True, help='Output directory path')
parser.add_argument('--txt-file', type=str, required=True, help='Test txt file path')
parser.add_argument('--method', type=str, default='EigenCAM', help='CAM method (default: EigenCAM)')
parser.add_argument('--target-layers', type=str, default='backbone.layer4', help='Target layer (default: backbone.layer4)')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
parser.add_argument('--clear-cache-interval', type=int, default=10, help='Clear GPU cache every N images (default: 10)')
parser.add_argument('--aug-smooth', action='store_true', help='Use augmentation smoothing')
args = parser.parse_args()

# Build command for batch processing
cmd = [
    "python", "tools/visualization/vis_cam.py",
    args.config,
    args.checkpoint,
    "--batch-file", args.txt_file,
    "--out-dir", args.out_dir,
    "--method", args.method,
    "--target-layers", args.target_layers,
    "--device", args.device,
    "--clear-cache-interval", str(args.clear_cache_interval),
]

if args.aug_smooth:
    cmd.append("--aug-smooth")

print("=" * 80)
print("Running batch CAM visualization with optimized memory management")
print("=" * 80)
print(f"Config: {args.config}")
print(f"Checkpoint: {args.checkpoint}")
print(f"Input file: {args.txt_file}")
print(f"Output dir: {args.out_dir}")
print(f"Method: {args.method}")
print(f"Device: {args.device}")
print(f"Memory clear interval: every {args.clear_cache_interval} images")
print("=" * 80)
print("\nCommand:", " ".join(cmd))
print()

# Run batch processing (single process, optimized memory)
subprocess.run(cmd, check=True)
