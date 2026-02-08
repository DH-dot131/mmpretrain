"""EVA-X Small finetuning for cervical spine stenosis with layerwise LR decay.

EVA-X Small (chest X-ray 사전학습) 계층별 학습률 감쇠 미세조정
- 의료 영상 도메인 (chest → cervical)
- 대규모 사전학습 데이터 (~520k images)
- Architecture: ViTEVA02 (RoPE + Sub-LN + SwiGLU)
- Layerwise LR: 초기 레이어(저수준 feature)는 낮은 LR, 나중 레이어(Classifier 근처)는 높은 LR
- Transfer learning 고려: 저수준 feature 보존 + task-specific adaptation
- 예상: ImageNet 대비 2-5% 향상

Key architectural features (from paper):
- RoPE (Rotational Position Embedding): Better position encoding than absolute/learned
- Sub-LN (Sublayer Normalization): Improved gradient flow
- SwiGLU: Better activation than GELU/ReLU in FFN

Transfer learning strategy:
- Layer decay rate 0.65: Earlier layers (edges, textures) preserved with lower LR
- Later layers (semantic features) adapted with higher LR for cervical spine task
- Medical domain knowledge from chest X-ray → cervical spine X-ray

Usage:
    # 1. Download EVA-X weights first
    python scripts/download_evax_weights.py --variant evax_small

    # 2. Train with EVA-X pretrained weights
    python mmpretrain/tools/train.py \
      mmpretrain/configs/csten/finetune/vit-small_evax-pretrain_layerwise-lr_8xb16_csten.py

    # 3. Monitor training
    tensorboard --logdir work_dirs/vit-small_evax-pretrain_layerwise-lr_8xb16_csten/
"""

_base_ = [
    '../../_base_/datasets/csten_v2_imbalanced_metrics.py',  # V2 dataset (C3-C6, 8 tasks)
    '../../_base_/schedules/csten_bs128_adamw.py',
    '../../_base_/csten_default_runtime.py'
]

# 커스텀 모듈 import (MultiTaskClsHeadV2)
custom_imports = dict(
    imports=['projects.spine_stenosis.models', 'projects.spine_stenosis.hooks'],
    allow_failed_imports=False
)

# 모델 설정 - EVA-X Small with ViTEVA02 backbone
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ViTEVA02',  # EVA02 architecture (same as EVA-X: RoPE + Sub-LN + SwiGLU)
        arch='s',  # Small: 384 hidden dim, 12 layers, 6 heads
        img_size=224,
        patch_size=16,  # EVA-X uses patch_size=16 unless specified otherwise
        drop_path_rate=0.1,  # Stochastic depth for regularization
        final_norm=True,  # Apply final layer norm
        out_type='avg_featmap',  # Average pooling over all tokens
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../checkpoints/evax/evax_small_mmpretrain.pth',  # Relative to mmpretrain/
            prefix='backbone.',
        )
    ),
    neck=None,  # ViT uses avg pooling, no additional neck needed
    head=dict(
        type='MultiTaskClsHeadV2',  # V2 head (8 tasks: C3-C6 only)
        num_classes=2,  # Binary: 0=absent/mild, 1=moderate/severe
        in_channels=384,  # ViT-Small hidden dimension
        loss=dict(
            type='FocalLoss',  # Focal loss for class imbalance
            gamma=2.0,  # Focusing parameter
            alpha=0.5,  # Class weight for positive class
            loss_weight=1.0,
            reduction='mean',
        ),
        # Task-level weights (prioritize central stenosis)
        task_weights=dict(
            C3_C4_central=2.0, C3_C4_foraminal=1.0,
            C4_C5_central=2.0, C4_C5_foraminal=1.0,
            C5_C6_central=2.0, C5_C6_foraminal=1.0,
            C6_C7_central=2.0, C6_C7_foraminal=1.0,
        ),
        # Auto-load class weights from annotations_v2/weights.json
        class_weights='auto_v2',
    ),
)

# LAYERWISE LEARNING RATE DECAY for transfer learning
# Strategy: Preserve low-level features, adapt high-level features
# - Earlier layers (near input): Lower LR → preserve medical imaging features from chest X-ray
# - Later layers (near classifier): Higher LR → adapt to cervical spine task
# - Layer decay rate 0.65: Typical range is 0.65-0.75 for ViT
#
# Example LR distribution for 12-layer ViT-Small:
#   Layer 0 (patch_embed):  base_lr * 0.65^11 ≈ base_lr * 0.009  (very low, preserve)
#   Layer 6 (middle):       base_lr * 0.65^5  ≈ base_lr * 0.116
#   Layer 11 (last):        base_lr * 0.65^0  = base_lr * 1.0     (full LR, adapt)
#   Classifier head:        base_lr * 1.0                         (full LR, task-specific)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,  # Base LR (applied to later layers and head)
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    constructor='LearningRateDecayOptimWrapperConstructor',  # Enable layerwise LR
    paramwise_cfg=dict(
        layer_decay_rate=0.65,  # Decay rate for layer-wise LR
        # Custom keys for components that should not have weight decay
        custom_keys={
            '.ln': dict(decay_mult=0.0),  # LayerNorm weight decay 제거
            '.bias': dict(decay_mult=0.0),  # Bias weight decay 제거
            'q_bias': dict(decay_mult=0.0),  # Query bias (if used)
            'v_bias': dict(decay_mult=0.0),  # Value bias (if used)
            '.cls_token': dict(decay_mult=0.0),  # CLS token weight decay 제거
            '.pos_embed': dict(decay_mult=0.0),  # Position embedding weight decay 제거
            '.gamma': dict(decay_mult=0.0),  # LayerScale gamma (if used in EVA)
        }
    )
)

# Learning rate scheduler
# - Warmup: 5 epochs (gradual LR increase to prevent training instability)
# - Cosine annealing: 95 epochs (smooth LR decay for better convergence)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,  # Start from base_lr * 1e-4
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=95,  # 95 epochs after warmup
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6  # Minimum LR at end of training
    )
]

# Training configuration
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

# Auto-scale LR based on batch size
# Formula: scaled_lr = base_lr * (actual_batch_size / base_batch_size)
auto_scale_lr = dict(base_batch_size=128)

# Work directory for this experiment
work_dir = '../work_dirs/vit-small_evax-pretrain_layerwise-lr_8xb16_csten'

# Visualization backends
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='c-spine-stenosis',
            name='vit-small_evax_chest-pretrain_layerwise-lr',
            tags=['finetune', 'evax', 'chest-pretrain', 'layerwise-lr', 'v2-dataset'],
            group='finetuning',
            notes=(
                'EVA-X Small (chest X-ray pretrained) with layerwise LR decay. '
                'Architecture: ViTEVA02 (RoPE + Sub-LN + SwiGLU). '
                'Transfer learning: chest → cervical spine. '
                'Expected: 2-5% improvement over ImageNet baseline.'
            )
        )
    )
]

visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends
)

# Runtime hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # Save every 5 epochs
        by_epoch=True,
        save_best='auto',  # Auto-save best model based on val metric
        max_keep_ckpts=3,  # Keep only 3 best checkpoints
        save_last=True  # Always save latest checkpoint
    ),
    logger=dict(type='LoggerHook', interval=50),  # Log every 50 iterations
)

# Reproducibility
randomness = dict(seed=42, deterministic=False)

# Training notes (for documentation)
# 1. Layerwise LR decay rate 0.65:
#    - Preserves low-level medical imaging features from chest X-ray pretraining
#    - Allows task-specific adaptation in later layers for cervical spine
# 2. Base LR 2e-3:
#    - Higher than typical ImageNet finetuning (usually 1e-3 to 5e-4)
#    - Justified because later layers need strong adaptation (chest → cervical)
# 3. Focal Loss with task weights:
#    - Addresses class imbalance in stenosis dataset
#    - Central stenosis weighted 2x (more clinically important)
# 4. Expected performance:
#    - Medical domain transfer (chest → cervical) may give 2-5% boost over ImageNet
#    - Large pretraining dataset (~520k) compensates for anatomical differences
