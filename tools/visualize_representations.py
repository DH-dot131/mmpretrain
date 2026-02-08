# Copyright (c) OpenMMLab. All rights reserved.
"""Unified tool for visualizing model representations.

Supports three visualization methods:
1. feature-extract: Extract and save backbone features
2. tsne: t-SNE 2D/3D visualization of feature space
3. cam: Class Activation Maps for spatial attention visualization
"""

import argparse
import os
import os.path as osp
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config, DictAction
from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmpretrain import FeatureExtractor
from mmpretrain.apis import get_model
from mmpretrain.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize model representations (feature extraction, t-SNE, CAM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features
  python tools/visualize_representations.py feature-extract \\
    configs/csten/resnet50_8xb16_csten.py \\
    work_dirs/resnet50_8xb16_csten/best_accuracy_top1_epoch_95.pth \\
    --images data/processed/images_png/*.png

  # t-SNE visualization
  python tools/visualize_representations.py tsne \\
    configs/csten/resnet50_8xb16_csten.py \\
    work_dirs/resnet50_8xb16_csten/best_accuracy_top1_epoch_95.pth \\
    --max-samples 100 --show

  # CAM visualization
  python tools/visualize_representations.py cam \\
    configs/csten/resnet50_8xb16_csten.py \\
    work_dirs/resnet50_8xb16_csten/best_accuracy_top1_epoch_95.pth \\
    --image data/processed/images_png/patient_001.png \\
    --method GradCAM
        """)

    # Common arguments
    parser.add_argument(
        'mode',
        choices=['feature-extract', 'tsne', 'cam'],
        help='Visualization mode: feature-extract, tsne, or cam')
    parser.add_argument('config', help='Model config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument(
        '--work-dir',
        help='Directory to save outputs (default: work_dirs/<config_name>/vis_<mode>_<timestamp>)')
    parser.add_argument(
        '--device',
        help='Device used for inference (default: auto-detect)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings, e.g., --cfg-options model.backbone.depth=50')

    # Feature extraction & t-SNE common arguments
    parser.add_argument(
        '--stage',
        choices=['backbone', 'neck', 'pre_logits'],
        default='backbone',
        help='Which model stage to extract features from (default: backbone)')

    # Feature extraction specific
    parser.add_argument(
        '--images',
        nargs='+',
        help='[feature-extract] Image file paths or glob patterns')
    parser.add_argument(
        '--batch-file',
        help='[feature-extract/cam] Text file with image paths (one per line)')
    parser.add_argument(
        '--save-format',
        choices=['npy', 'pt', 'npz'],
        default='npy',
        help='[feature-extract] Format to save features (default: npy)')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='[feature-extract] Create PNG grid visualization of features')
    parser.add_argument(
        '--grid-cols',
        type=int,
        default=8,
        help='[feature-extract] Number of columns in feature grid (default: 8)')
    parser.add_argument(
        '--max-vis-samples',
        type=int,
        default=64,
        help='[feature-extract] Maximum samples to visualize in grid (default: 64)')

    # t-SNE specific arguments
    parser.add_argument(
        '--test-cfg',
        help='[tsne] Config file for test dataloader (if different from main config)')
    parser.add_argument(
        '--task-idx',
        type=int,
        help='[tsne] Specific task index to visualize (0-13 for 14 tasks). '
             'If not set, uses combined multi-task labels')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='[tsne] Maximum samples per class (default: 100)')
    parser.add_argument(
        '--n-components',
        type=int,
        default=2,
        choices=[2, 3],
        help='[tsne] Number of dimensions (2D or 3D, default: 2)')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='[tsne] t-SNE perplexity parameter (default: 30.0)')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=200.0,
        help='[tsne] t-SNE learning rate (default: 200.0)')
    parser.add_argument(
        '--n-iter',
        type=int,
        default=1000,
        help='[tsne] Maximum iterations (default: 1000)')
    parser.add_argument(
        '--show',
        action='store_true',
        help='[tsne] Display the plot interactively')
    parser.add_argument(
        '--legend',
        action='store_true',
        help='[tsne] Show legend with class names')

    # CAM specific arguments
    parser.add_argument(
        '--image',
        help='[cam] Single image path for CAM visualization')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='[cam] CAM method: GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM (default: GradCAM)')
    parser.add_argument(
        '--target-layers',
        nargs='+',
        help='[cam] Target layer names (default: auto-detect last conv layer)')
    parser.add_argument(
        '--target-task',
        type=int,
        help='[cam] Target task index (0-13) for multi-task models')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='[cam] Apply eigen smoothing to reduce noise')
    parser.add_argument(
        '--aug-smooth',
        action='store_true',
        help='[cam] Apply test-time augmentation smoothing')

    args = parser.parse_args()
    return args


def setup_work_dir(args):
    """Setup working directory for outputs."""
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        config_name = osp.splitext(osp.basename(args.config))[0]
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        work_dir = osp.join(
            './work_dirs',
            config_name,
            f'vis_{args.mode}_{timestamp}'
        )

    mkdir_or_exist(osp.abspath(work_dir))
    return work_dir, cfg


def create_feature_grid_visualization(features, image_names, work_dir, grid_cols, max_samples, logger):
    """Create a grid visualization of extracted features.

    Args:
        features: numpy array of shape (N, D) where N is number of samples and D is feature dimension
        image_names: list of image filenames
        work_dir: directory to save visualization
        grid_cols: number of columns in grid
        max_samples: maximum number of samples to visualize
        logger: logger instance
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning('scikit-learn not installed. Skipping visualization. Install with: pip install scikit-learn')
        return

    # Limit number of samples
    n_samples = min(len(features), max_samples)
    features_subset = features[:n_samples]
    names_subset = image_names[:n_samples]

    logger.info(f'Visualizing {n_samples} samples in grid...')

    # Reduce features to 3D for RGB visualization using PCA
    if features_subset.shape[1] > 3:
        logger.info(f'Reducing features from {features_subset.shape[1]}D to 3D using PCA...')
        pca = PCA(n_components=3)
        features_reduced = pca.fit_transform(features_subset)
        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f'PCA explained variance: {explained_var:.2%}')
    else:
        features_reduced = features_subset

    # Normalize to [0, 1] for visualization
    feat_min = features_reduced.min(axis=0)
    feat_max = features_reduced.max(axis=0)
    features_norm = (features_reduced - feat_min) / (feat_max - feat_min + 1e-8)

    # Create grid
    grid_rows = int(np.ceil(n_samples / grid_cols))

    # Create figure
    fig_width = grid_cols * 2  # 2 inches per column
    fig_height = grid_rows * 2.5  # 2.5 inches per row (extra space for labels)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier indexing
    if grid_rows == 1:
        axes = axes.reshape(1, -1)
    elif grid_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(n_samples):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]

        # Create RGB visualization from 3D features
        rgb = features_norm[idx]
        rgb_square = np.tile(rgb, (50, 50, 1))  # Create 50x50 colored square

        ax.imshow(rgb_square)
        ax.set_title(f'{names_subset[idx][:15]}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_samples, grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        axes[row, col].axis('off')

    plt.suptitle(f'Feature Visualization Grid (PCA to RGB, {n_samples} samples)', fontsize=14)
    plt.tight_layout()

    # Save
    grid_path = osp.join(work_dir, 'feature_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    logger.info(f'Feature grid visualization saved to: {grid_path}')
    plt.close()

    # Also create a feature heatmap visualization
    logger.info('Creating feature heatmap...')
    fig, ax = plt.subplots(figsize=(12, max(6, n_samples * 0.3)))

    # Show features as heatmap (samples x dimensions)
    im = ax.imshow(features_norm, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Feature Dimension (RGB from PCA)', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'Feature Heatmap ({n_samples} samples, 3D features)', fontsize=14)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Normalized Feature Value')

    # Add sample labels on y-axis (if not too many)
    if n_samples <= 50:
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([name[:20] for name in names_subset], fontsize=8)
    else:
        ax.set_yticks(range(0, n_samples, max(1, n_samples // 20)))

    plt.tight_layout()
    heatmap_path = osp.join(work_dir, 'feature_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    logger.info(f'Feature heatmap saved to: {heatmap_path}')
    plt.close()


def feature_extract_mode(args, cfg, work_dir, logger):
    """Extract and save backbone features."""
    if not args.images and not args.batch_file:
        raise ValueError('Either --images or --batch-file is required for feature-extract mode')

    logger.info(f'Extracting features using stage: {args.stage}')

    # Initialize feature extractor
    device = args.device or get_device()
    extractor = FeatureExtractor(
        args.config,
        pretrained=args.checkpoint,
        device=device
    )
    logger.info(f'Feature extractor initialized on {device}')

    # Process images
    from glob import glob
    import os

    logger.info(f'Current working directory: {os.getcwd()}')

    image_paths = []

    # Load from batch file if provided
    if args.batch_file:
        batch_file_path = args.batch_file if osp.isabs(args.batch_file) else osp.join(os.getcwd(), args.batch_file)
        if not osp.exists(batch_file_path):
            raise ValueError(f'Batch file not found: {args.batch_file} (checked: {batch_file_path})')

        logger.info(f'Loading image paths from batch file: {batch_file_path}')
        with open(batch_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Convert to absolute path if relative
                    if not osp.isabs(line):
                        abs_path = osp.join(os.getcwd(), line)
                    else:
                        abs_path = line

                    if osp.exists(abs_path):
                        image_paths.append(abs_path)
                    else:
                        logger.warning(f'Image not found: {line} (checked: {abs_path})')

        logger.info(f'Loaded {len(image_paths)} image paths from batch file')

    # Load from --images patterns if provided
    elif args.images:
        for pattern in args.images:
            if '*' in pattern:
                # Convert to absolute path if relative
                if not osp.isabs(pattern):
                    abs_pattern = osp.join(os.getcwd(), pattern)
                    logger.info(f'Searching pattern: {abs_pattern}')
                    matched = glob(abs_pattern)
                else:
                    logger.info(f'Searching pattern: {pattern}')
                    matched = glob(pattern)

                if matched:
                    image_paths.extend(matched)
                    logger.info(f'  Found {len(matched)} images')
                else:
                    logger.warning(f'  No images matched pattern: {pattern}')
            else:
                # Single file path
                if not osp.isabs(pattern):
                    abs_path = osp.join(os.getcwd(), pattern)
                else:
                    abs_path = pattern

                if osp.exists(abs_path):
                    image_paths.append(abs_path)
                else:
                    logger.warning(f'Image not found: {pattern} (checked: {abs_path})')

    logger.info(f'Processing {len(image_paths)} images...')

    if len(image_paths) == 0:
        logger.error('No images found! Please check:')
        logger.error(f'  1. Current directory: {os.getcwd()}')
        logger.error(f'  2. Image patterns/batch file: {args.images or args.batch_file}')
        logger.error(f'  3. Try using absolute paths or run from project root')
        raise ValueError('No images to process')

    features_list = []
    image_names = []

    for img_path in image_paths:
        if not osp.exists(img_path):
            logger.warning(f'Image not found: {img_path}')
            continue

        # Extract features
        feat = extractor(img_path, stage=args.stage)[0]

        # Handle tuple/list of features (backbone stage returns multiple layers)
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]  # Use last layer features

        # Handle different feature shapes
        if feat.ndim == 4:  # (N, C, H, W) - shouldn't happen but just in case
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze()
        elif feat.ndim == 3:  # (C, H, W)
            feat = F.adaptive_avg_pool2d(feat.unsqueeze(0), 1).squeeze()
        elif feat.ndim == 2:  # (L, C) for transformers
            feat = feat.mean(0)
        # else: feat.ndim == 1, already a feature vector

        features_list.append(feat.cpu().numpy())
        image_names.append(osp.basename(img_path))

        logger.info(f'Extracted: {osp.basename(img_path)} -> shape {feat.shape}')

    features = np.array(features_list)
    logger.info(f'Final feature matrix shape: {features.shape}')

    # Save features
    if args.save_format == 'npy':
        save_path = osp.join(work_dir, 'features.npy')
        np.save(save_path, features)
        # Save metadata
        meta_path = osp.join(work_dir, 'image_names.txt')
        with open(meta_path, 'w') as f:
            f.write('\n'.join(image_names))
    elif args.save_format == 'pt':
        save_path = osp.join(work_dir, 'features.pt')
        torch.save({
            'features': torch.from_numpy(features),
            'image_names': image_names
        }, save_path)
    elif args.save_format == 'npz':
        save_path = osp.join(work_dir, 'features.npz')
        np.savez(save_path, features=features, image_names=image_names)

    logger.info(f'Features saved to: {save_path}')
    logger.info(f'Image names saved to: {osp.join(work_dir, "image_names.txt")}')

    # Create PNG grid visualization if requested
    if args.visualize:
        logger.info('Creating feature grid visualization...')
        create_feature_grid_visualization(
            features, image_names, work_dir, args.grid_cols, args.max_vis_samples, logger
        )


def tsne_mode(args, cfg, work_dir, logger):
    """t-SNE visualization of feature space."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError('Please install scikit-learn: pip install scikit-learn')

    import rich.progress as progress

    logger.info(f't-SNE visualization using stage: {args.stage}')

    # Build model
    device = args.device or get_device()
    model = get_model(cfg, args.checkpoint, device=device)
    logger.info('Model loaded.')

    # Build dataset
    if args.test_cfg is not None:
        dataloader_cfg = Config.fromfile(args.test_cfg).get('test_dataloader')
    elif 'test_dataloader' not in cfg:
        raise ValueError('No test_dataloader in config. Use --test-cfg to specify.')
    else:
        dataloader_cfg = cfg.get('test_dataloader')

    dataset = DATASETS.build(dataloader_cfg.pop('dataset'))

    # For multi-task dataset, extract task info
    metainfo = dataset.metainfo
    if args.task_idx is not None:
        tasks = metainfo.get('tasks', [])
        if args.task_idx >= len(tasks):
            raise ValueError(f'task_idx {args.task_idx} out of range (0-{len(tasks)-1})')
        task_name = tasks[args.task_idx]
        logger.info(f'Visualizing task: {task_name} (index {args.task_idx})')
    else:
        logger.info('Visualizing combined multi-task labels')

    # Sample subset
    subset_idx_list = []
    counter = defaultdict(int)

    for i in range(len(dataset)):
        data_info = dataset.get_data_info(i)

        # Get label for visualization
        if args.task_idx is not None:
            # Single task
            gt_label = data_info['gt_label'].get(task_name, 0)
        else:
            # Multi-task: use any positive label
            labels = list(data_info['gt_label'].values())
            gt_label = 1 if any(labels) else 0

        if counter[gt_label] < args.max_samples:
            subset_idx_list.append(i)
            counter[gt_label] += 1

    dataset.get_subset_(subset_idx_list)
    logger.info(f'Using {len(subset_idx_list)} samples for t-SNE')
    logger.info(f'Class distribution: {dict(counter)}')

    dataloader_cfg.dataset = dataset
    dataloader_cfg.setdefault('collate_fn', dict(type='default_collate'))
    dataloader = Runner.build_dataloader(dataloader_cfg)

    # Extract features
    features = []
    labels = []

    for data in progress.track(dataloader, description='Extracting features...'):
        with torch.no_grad():
            data = model.data_preprocessor(data)
            batch_inputs, batch_data_samples = data['inputs'], data['data_samples']

            # Extract features
            extract_args = {'stage': args.stage} if args.stage else {}
            batch_features = model.extract_feat(batch_inputs, **extract_args)

            # Post-process features
            if isinstance(batch_features, (list, tuple)):
                batch_features = batch_features[-1]  # Use last layer

            if batch_features.ndim == 4:  # (N, C, H, W)
                batch_features = F.adaptive_avg_pool2d(batch_features, 1).squeeze(-1).squeeze(-1)
            elif batch_features.ndim == 3:  # (N, L, C)
                batch_features = batch_features.mean(1)

            features.append(batch_features.cpu().numpy())

            # Get labels
            for sample in batch_data_samples:
                if args.task_idx is not None:
                    label = sample.gt_label.get(task_name, 0)
                else:
                    label_values = list(sample.gt_label.values())
                    label = 1 if any(label_values) else 0
                labels.append(label)

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    logger.info(f'Feature shape: {features.shape}, Labels shape: {labels.shape}')

    # Save raw features
    np.save(osp.join(work_dir, 'features.npy'), features)
    np.save(osp.join(work_dir, 'labels.npy'), labels)

    # Run t-SNE
    logger.info('Running t-SNE...')
    tsne = TSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_state=42
    )
    features_tsne = tsne.fit_transform(features)

    # Normalize to [0, 1]
    feat_min, feat_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - feat_min) / (feat_max - feat_min + 1e-8)

    # Save t-SNE results
    np.save(osp.join(work_dir, 'tsne_features.npy'), features_norm)

    # Visualize
    if args.n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                features_norm[mask, 0],
                features_norm[mask, 1],
                c=[color],
                label=f'Class {label}',
                alpha=0.7,
                s=30
            )

        if args.legend:
            ax.legend(loc='best')

        title = f't-SNE Visualization ({args.stage})'
        if args.task_idx is not None:
            title += f' - {task_name}'
        ax.set_title(title)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    else:  # 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                features_norm[mask, 0],
                features_norm[mask, 1],
                features_norm[mask, 2],
                c=[color],
                label=f'Class {label}',
                alpha=0.7,
                s=30
            )

        if args.legend:
            ax.legend(loc='best')

        title = f't-SNE Visualization ({args.stage})'
        if args.task_idx is not None:
            title += f' - {task_name}'
        ax.set_title(title)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')

    plt.tight_layout()
    save_path = osp.join(work_dir, 'tsne_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f't-SNE plot saved to: {save_path}')

    if args.show:
        plt.show()


def cam_mode(args, cfg, work_dir, logger):
    """Class Activation Map visualization."""
    try:
        import pytorch_grad_cam as cam
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
    except ImportError:
        raise ImportError('Please install grad-cam: pip install "grad-cam>=1.3.6"')

    import cv2
    from mmcv.transforms import Compose
    from mmpretrain.registry import TRANSFORMS

    # Method mapping
    METHOD_MAP = {
        'gradcam': cam.GradCAM,
        'gradcam++': cam.GradCAMPlusPlus,
        'gradcamplusplus': cam.GradCAMPlusPlus,
        'scorecam': cam.ScoreCAM,
        'eigencam': cam.EigenCAM,
        'layercam': cam.LayerCAM,
        'xgradcam': cam.XGradCAM,
        'fullgrad': cam.FullGrad,
    }

    method_key = args.method.lower()
    if method_key not in METHOD_MAP:
        raise ValueError(f'Unknown CAM method: {args.method}. '
                        f'Available: {list(METHOD_MAP.keys())}')

    cam_method = METHOD_MAP[method_key]
    logger.info(f'Using CAM method: {args.method}')

    # Load model
    device = args.device or get_device()
    model = get_model(cfg, args.checkpoint, device=device)
    model.eval()
    logger.info('Model loaded.')

    # Auto-detect target layers if not specified
    if not args.target_layers:
        # Find last conv/norm layer in backbone
        target_layers = []
        for name, module in model.backbone.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                target_layers = [module]

        if not target_layers:
            raise ValueError('Could not auto-detect target layers. Please specify --target-layers')
        logger.info(f'Auto-detected target layer: {target_layers[0].__class__.__name__}')
    else:
        # Get layers by name
        target_layers = []
        for layer_name in args.target_layers:
            layer = dict(model.named_modules()).get(layer_name)
            if layer is None:
                raise ValueError(f'Layer not found: {layer_name}')
            target_layers.append(layer)
        logger.info(f'Using target layers: {args.target_layers}')

    # Initialize CAM
    cam_extractor = cam_method(
        model=model,
        target_layers=target_layers,
        use_cuda=(device != 'cpu')
    )

    # Prepare image pipeline
    test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    from mmpretrain.datasets import remove_transform
    test_pipeline_cfg = remove_transform(test_pipeline_cfg, 'LoadImageFromFile')
    test_pipeline = Compose([TRANSFORMS.build(t) for t in test_pipeline_cfg])

    # Get images to process
    import os
    if args.image:
        image_paths = [args.image]
    elif args.batch_file:
        batch_file_path = args.batch_file if osp.isabs(args.batch_file) else osp.join(os.getcwd(), args.batch_file)
        if not osp.exists(batch_file_path):
            raise ValueError(f'Batch file not found: {args.batch_file} (checked: {batch_file_path})')
        with open(batch_file_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError('Either --image or --batch-file is required for CAM mode')

    logger.info(f'Current working directory: {os.getcwd()}')
    logger.info(f'Processing {len(image_paths)} images...')

    processed_count = 0
    for img_path in image_paths:
        # Convert to absolute path if relative
        if not osp.isabs(img_path):
            abs_img_path = osp.join(os.getcwd(), img_path)
        else:
            abs_img_path = img_path

        if not osp.exists(abs_img_path):
            logger.warning(f'Image not found: {img_path} (checked: {abs_img_path})')
            continue

        img_path = abs_img_path  # Use absolute path
        processed_count += 1

        # Load and preprocess image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        # Prepare input
        data = dict(img=img_rgb, img_shape=img_rgb.shape[:2], ori_shape=img_rgb.shape[:2])
        data = test_pipeline(data)
        inputs = data['inputs'].unsqueeze(0).to(device)

        # Generate CAM
        with torch.no_grad():
            if args.target_task is not None:
                # For multi-task, need custom target
                # This is a simplified version - may need adjustment
                grayscale_cam = cam_extractor(
                    input_tensor=inputs,
                    eigen_smooth=args.eigen_smooth,
                    aug_smooth=args.aug_smooth
                )
            else:
                grayscale_cam = cam_extractor(
                    input_tensor=inputs,
                    eigen_smooth=args.eigen_smooth,
                    aug_smooth=args.aug_smooth
                )

        # Overlay CAM on image
        grayscale_cam = grayscale_cam[0, :]

        # Resize to original image size
        h, w = img_rgb.shape[:2]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))

        cam_image = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)

        # Save result
        img_name = osp.splitext(osp.basename(img_path))[0]
        save_path = osp.join(work_dir, f'{img_name}_cam.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        logger.info(f'CAM saved: {save_path}')

        # Also save heatmap only
        heatmap_path = osp.join(work_dir, f'{img_name}_heatmap.jpg')
        heatmap_colored = cv2.applyColorMap(
            (grayscale_cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imwrite(heatmap_path, heatmap_colored)

    if processed_count == 0:
        logger.error('No images were processed! Please check:')
        logger.error(f'  1. Current directory: {os.getcwd()}')
        logger.error(f'  2. Image paths: {image_paths[:5]}...' if len(image_paths) > 5 else f'  2. Image paths: {image_paths}')
        logger.error(f'  3. Try using absolute paths or run from project root')
        raise ValueError('No images processed')

    logger.info(f'{processed_count} CAM visualizations saved to: {work_dir}')


def main():
    args = parse_args()

    # Setup
    work_dir, cfg = setup_work_dir(args)

    # Initialize logger
    log_file = osp.join(work_dir, f'{args.mode}.log')
    logger = MMLogger.get_instance(
        'mmpretrain',
        logger_name='mmpretrain',
        log_file=log_file,
        log_level='INFO'
    )

    logger.info(f'Mode: {args.mode}')
    logger.info(f'Config: {args.config}')
    logger.info(f'Checkpoint: {args.checkpoint}')
    logger.info(f'Working directory: {work_dir}')

    # Execute mode
    if args.mode == 'feature-extract':
        feature_extract_mode(args, cfg, work_dir, logger)
    elif args.mode == 'tsne':
        tsne_mode(args, cfg, work_dir, logger)
    elif args.mode == 'cam':
        cam_mode(args, cfg, work_dir, logger)

    logger.info('Done!')


if __name__ == '__main__':
    main()
