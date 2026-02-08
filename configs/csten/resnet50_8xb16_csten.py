"""ResNet-50 baseline with Enhanced TensorBoard logging.

경추 척추관 협착증 분류를 위한 ResNet-50 베이스라인 모델 (향상된 TensorBoard 로깅).
- Backbone: ResNet-50 (ImageNet pretrained)
- Head: Multi-task classification head (21 tasks)
- Dataset: Lateral X-ray images
- Training: 100 epochs with AdamW optimizer
- Logging: Enhanced TensorBoard with gradient norms, weight histograms, etc.

Usage:
    # 학습
    python mmpretrain/tools/train.py mmpretrain/configs/csten/resnet50_8xb16_csten_enhanced_tb.py

    # 테스트
    python mmpretrain/tools/test.py mmpretrain/configs/csten/resnet50_8xb16_csten_enhanced_tb.py work_dirs/resnet50_8xb16_csten_enhanced_tb/best_accuracy_top1_epoch_*.pth

    # TensorBoard 시작
    tensorboard --logdir work_dirs/resnet50_8xb16_csten_enhanced_tb
"""

_base_ = [
    '../_base_/datasets/csten_lateral_bs16.py',
    '../_base_/schedules/csten_bs128_adamw.py',
    '../_base_/default_runtime.py'
]

# 커스텀 모듈 import (models + hooks)
custom_imports = dict(
    imports=['projects.spine_stenosis.models', 'projects.spine_stenosis.hooks'],
    allow_failed_imports=False
)

# 모델 설정
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
        num_classes=2,  # Binary: 0=absent/mild, 1=moderate/severe
        in_channels=2048,  # ResNet-50의 출력 채널 수
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        task_weights=None,
    ),
)

# 런타임 설정
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

# Enhanced TensorBoard Hook 추가
custom_hooks = [
    dict(
        type='EnhancedTensorBoardHook',
        log_grad_norm=True,           # Gradient norm 로깅
        log_weight_hist=True,         # Weight histogram 로깅
        log_val_metrics=True,         # Validation metrics 로깅
        log_lr=True,                  # Learning rate 로깅
        log_model_stats=True,         # 모델 통계 로깅
        grad_norm_interval=100,       # Gradient norm 로깅 간격 (iterations)
        weight_hist_interval=1,       # Weight histogram 로깅 간격 (epochs)
        max_layers_to_log=50,         # 개별 로깅할 최대 레이어 수
    )
]

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

work_dir = '../work_dirs/resnet50_8xb16_csten_enhanced_tb'
