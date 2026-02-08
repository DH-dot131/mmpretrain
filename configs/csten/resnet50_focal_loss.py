"""ResNet-50 with Focal Loss for imbalanced cervical spine stenosis classification.

Focal Loss automatically handles class imbalance by:
- Downweighting easy examples (well-classified samples)
- Focusing on hard examples (misclassified or uncertain samples)

Parameters:
- gamma: Focusing parameter (higher = more focus on hard examples). Default: 2.0
- alpha: Class balancing parameter (0.25 for minority class). Default: 0.25

Expected improvement: 15-25% F1-score increase for minority class vs baseline.

Usage:
    python mmpretrain/tools/train.py mmpretrain/configs/csten/resnet50_focal_loss.py
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
        type='MultiTaskClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(
            type='FocalLoss',
            gamma=2.0,        # Focus on hard examples (higher = more aggressive)
            alpha=0.25,       # Weight for minority class (class 1)
            loss_weight=1.0,
            reduction='mean',
        ),
        task_weights=None,  # No task-level reweighting (yet)
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

work_dir = '../work_dirs/resnet50_focal_loss'
