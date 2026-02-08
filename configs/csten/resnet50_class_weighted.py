"""ResNet-50 with per-task class weighting and task-level reweighting.

This configuration implements comprehensive imbalance handling:
1. Per-task class weights (automatically loaded from analysis)
2. Task-level reweighting (upweight severely imbalanced tasks)
3. Task exclusion (C1-C2 tasks with zero positive samples)

Expected improvement: 25-40% F1-score increase for minority class vs baseline.

Usage:
    python mmpretrain/tools/train.py mmpretrain/configs/csten/resnet50_class_weighted.py
"""

_base_ = [
    '../_base_/datasets/csten_lateral_bs16.py',
    '../_base_/schedules/csten_bs128_adamw.py',
    '../_base_/default_runtime.py'
]

# Custom module imports
custom_imports = dict(
    imports=['projects.spine_stenosis.models'],
    allow_failed_imports=False
)

# Task-level weights (upweight severely imbalanced tasks)
# Based on class distribution analysis
task_weights = {
    # Exclude C1-C2 (no positive samples in training set)
    'C1_C2_central': 0.0,
    'C1_C2_foraminal': 0.0,

    # Severely imbalanced central stenosis tasks (ratio > 100:1)
    'C2_C3_central': 10.0,    # 130:1 ratio, only 3 positive samples
    'C7_T1_central': 10.0,    # 130:1 ratio, only 3 positive samples

    # Very imbalanced foraminal task
    'C2_C3_foraminal': 5.0,   # 43:1 ratio, only 9 positive samples

    # Moderately imbalanced central stenosis (ratio 10-25:1)
    'C3_C4_central': 3.0,     # 18:1 ratio
    'C4_C5_central': 3.0,     # 19:1 ratio
    'C5_C6_central': 3.0,     # 14:1 ratio
    'C6_C7_central': 3.0,     # 24:1 ratio

    # Moderately imbalanced foraminal task
    'C7_T1_foraminal': 3.0,   # 18:1 ratio

    # Relatively balanced foraminal tasks (ratio < 5:1)
    'C3_C4_foraminal': 1.0,   # 3.3:1 ratio
    'C4_C5_foraminal': 1.0,   # 2.6:1 ratio
    'C5_C6_foraminal': 1.0,   # 1.0:1 ratio (nearly balanced)
    'C6_C7_foraminal': 1.0,   # 1.6:1 ratio
}

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
        type='MultiTaskClsHeadWeighted',  # Enhanced head with class weighting
        num_classes=2,
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            reduction='mean',
            # Class weights will be loaded per-task from JSON
        ),
        task_weights=task_weights,      # Task-level reweighting
        class_weights='auto',            # Auto-load from class_weights.json
    ),
)

# Runtime settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        by_epoch=True,
        save_best='auto',
        max_keep_ckpts=3,
        save_last=True
    ),
    logger=dict(type='LoggerHook', interval=50),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=1,
)

randomness = dict(seed=42, deterministic=False)

work_dir = '../work_dirs/resnet50_class_weighted'
