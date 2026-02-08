"""ResNet-50 V2: Filtered Dataset (C3-C6) + Focal Loss + Class/Task Weighting + Layerwise LR + Enhanced TensorBoard.

This configuration implements the complete class imbalance handling strategy:
1. **Dataset Filtering**: Uses annotations_v2 (C3-C6 only, excludes C1_C2/C2_C3/C7_T1)
2. **Focal Loss**: gamma=2.0, alpha=0.25 (automatic hard example mining)
3. **Class Weighting**: Per-task inverse frequency weights from training set
4. **Task Weighting**: Focus on foraminal stenosis (2.5x), central stenosis as auxiliary (0.3x)
5. **Layerwise Learning Rate**: Backbone stages use different LR (0.3x-1.2x), Head (3x) adapts to task (stabilized)
6. **Enhanced TensorBoard**: Gradient norms, weight histograms, per-task metrics

Dataset Statistics (annotations_v2):
- Total tasks: 8 (down from 14)
- Overall imbalance: 4.0:1 (improved from 7.3:1)
- Central stenosis: 13-24:1 ratio (auxiliary task, task_weight=0.3)
- Foraminal stenosis: 1-3.3:1 ratio (primary target, task_weight=2.5)

Expected Performance:
- Baseline (no imbalance handling): F1 ~30-40%, Recall ~20-30%
- With this config: F1 ~65-75%, Recall ~70-85%
- Improvement: +35-45% F1-score vs baseline
- Layerwise LR: Early stages (0.3x-0.6x) preserve pretrained features, later stages (1.0x-1.2x) adapt, Head (3x) learns task-specific features
- Stabilization: Gradient clipping (max_norm=1.0), longer warmup (10 epochs)

Usage:
    # Training
    python mmpretrain/tools/train.py mmpretrain/configs/csten/resnet50_8xb16_csten_v2_focal_weighted_layerlr.py

    # Testing
    python mmpretrain/tools/test.py \
        mmpretrain/configs/csten/resnet50_8xb16_csten_v2_focal_weighted_layerlr.py \
        work_dirs/resnet50_v2_focal_weighted_layerlr/best_*.pth

    # TensorBoard monitoring
    tensorboard --logdir work_dirs/resnet50_v2_focal_weighted_layerlr
"""

_base_ = [
    '../_base_/datasets/csten_v2_imbalanced_metrics.py',  # V2 dataset with comprehensive metrics
    '../_base_/schedules/csten_bs128_adamw.py',
    '../_base_/csten_default_runtime.py'  # Project-specific runtime (no save_best)
]

# Custom module imports (models + hooks)
custom_imports = dict(
    imports=['projects.spine_stenosis.models', 'projects.spine_stenosis.hooks'],
    allow_failed_imports=False
)

# Load task weights from annotations_v2
# Note: Loaded inline to avoid pickle issues - file is closed immediately after reading
import json as _json_module
import os as _os_module
# Try multiple paths since __file__ may not be available in MMEngine config loader
_weights_paths = [
    '../data/processed/annotations_v2/weights.json',  # From mmpretrain directory
    'data/processed/annotations_v2/weights.json',    # From project root
]
if '__file__' in globals():
    _config_dir = _os_module.path.dirname(_os_module.path.abspath(__file__))
    _weights_paths.insert(0, _os_module.path.normpath(
        _os_module.path.join(_config_dir, '../data/processed/annotations_v2/weights.json')))
_weights_path = None
for _path in _weights_paths:
    if _os_module.path.exists(_path):
        _weights_path = _path
        break
if _weights_path is None:
    raise FileNotFoundError(f'weights.json not found in any of: {_weights_paths}')
with open(_weights_path, 'r', encoding='utf-8') as _weights_file:
    weights_data = _json_module.load(_weights_file)
    base_task_weights = weights_data['task_weights']
    base_class_weights = weights_data['class_weights']

# Adjust task weights: Focus on foraminal stenosis, reduce central stenosis weight
# Central stenosis is auxiliary (helps foraminal prediction), foraminal is primary target
task_weights = {}
for task_name, weight in base_task_weights.items():
    if 'central' in task_name:
        task_weights[task_name] = 0.3  # Reduce central stenosis weight (2.0 -> 0.3)
    else:  # foraminal
        task_weights[task_name] = 2.5  # Increase foraminal stenosis weight (1.0 -> 2.5)

# Adjust class weights: Increase positive class weight to reduce false negatives
# Multiply positive class (index 1) weight by 1.5-2.0x to encourage more positive predictions
class_weights = {}
for task_name, weights in base_class_weights.items():
    adjusted_weights = weights.copy()
    # Increase positive class weight (index 1) to reduce false negatives
    if 'foraminal' in task_name:
        # Foraminal tasks: more aggressive increase
        adjusted_weights[1] = weights[1] * 2.0  # 2x increase for foraminal
    else:
        # Central tasks: moderate increase (since it's auxiliary)
        adjusted_weights[1] = weights[1] * 1.5  # 1.5x increase for central
    class_weights[task_name] = adjusted_weights

# Clean up to avoid pickle issues
del _json_module, _os_module, _weights_path, _weights_paths, _weights_file, weights_data, base_task_weights, base_class_weights
if '__file__' in globals():
    del _config_dir

# Model configuration
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone.'),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskClsHeadV2',  # V2 head (8 tasks only)
        num_classes=2,  # Binary: 0=absent/mild, 1=moderate/severe
        in_channels=2048,  # ResNet-50 output channels
        loss=dict(
            type='FocalLoss',
            gamma=2.0,        # Focus on hard examples
            alpha=0.5,       # Increased minority class weight (0.25 -> 0.5) to reduce false negatives
            loss_weight=1.0,
            reduction='mean',
        ),
        task_weights=task_weights,      # Task-level reweighting (foraminal=2.5x, central=0.3x)
        class_weights=class_weights,   # Adjusted class weights (increased positive class weight)
    ),
)

# Layerwise Learning Rate Configuration (Stabilized Version)
# Strategy: More conservative LR multipliers + Gradient clipping for stable training
# - Stage 0 (conv1, bn1): lr_mult=0.3 (increased from 0.1, preserve low-level features)
# - Stage 1 (layer1): lr_mult=0.6 (increased from 0.5, preserve early features)
# - Stage 2 (layer2): lr_mult=0.8 (increased from 0.75, medium-low)
# - Stage 3 (layer3): lr_mult=1.0 (base LR)
# - Stage 4 (layer4): lr_mult=1.2 (decreased from 1.5, adapt semantic features)
# - Head: lr_mult=3.0 (decreased from 10.0, task-specific adaptation)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,  # Base learning rate (from base schedule)
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    # Gradient clipping 추가 (gradient explosion 방지)
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,  # Normalization layer는 weight decay 적용 안함
        bias_decay_mult=0.0,  # Bias는 weight decay 적용 안함
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
            # Backbone 초기 레이어 (더 높은 LR로 조정)
            'backbone.conv1': dict(lr_mult=0.3),  # Initial conv: base_lr * 0.3 (0.1 -> 0.3)
            'backbone.bn1': dict(lr_mult=0.3),    # Initial BN: base_lr * 0.3 (0.1 -> 0.3)
            # Backbone 각 stage별 LR (더 보수적으로 조정)
            'backbone.layer1': dict(lr_mult=0.6),   # Early stage: base_lr * 0.6 (0.5 -> 0.6)
            'backbone.layer2': dict(lr_mult=0.8),  # Mid-early stage: base_lr * 0.8 (0.75 -> 0.8)
            'backbone.layer3': dict(lr_mult=1.0),   # Mid stage: base_lr * 1.0 (유지)
            'backbone.layer4': dict(lr_mult=1.2),   # Late stage: base_lr * 1.2 (1.5 -> 1.2)
            # Head에 더 보수적인 learning rate 적용
            'head': dict(lr_mult=3.0),  # Head learning rate = base_lr * 3.0 (10.0 -> 3.0)
        }
    )
)

# Enhanced TensorBoard Hook (added to base runtime hooks)
custom_hooks = [
    dict(
        type='EnhancedTensorBoardHook',
        log_grad_norm=True,           # Gradient norm logging
        log_weight_hist=True,         # Weight histogram logging
        log_val_metrics=True,         # Validation metrics logging
        log_lr=True,                  # Learning rate logging
        log_model_stats=True,         # Model statistics logging
        grad_norm_interval=100,       # Gradient norm logging interval (iterations)
        weight_hist_interval=1,       # Weight histogram logging interval (epochs)
        max_layers_to_log=50,         # Max layers to log individually
    )
]

# Learning rate scheduler (override base schedule for longer warmup)
# - Longer warmup (10 epochs) for stable training with layerwise LR
# - Cosine annealing for remaining 90 epochs
param_scheduler = [
    # Warmup (더 긴 warmup으로 안정화)
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=10,  # 5 -> 10 epoch warmup (더 안정적인 학습)
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=90,  # 100 - 10 (warmup)
        by_epoch=True,
        begin=10,
        end=100,
    )
]

# Note: default_hooks, vis_backends, and visualizer are defined in csten_default_runtime.py

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=1,
)

randomness = dict(seed=42, deterministic=False)

work_dir = '../work_dirs/resnet50_v2_focal_weighted_layerlr'