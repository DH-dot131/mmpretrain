"""EVA-X Small STABLE + ENHANCED AUGMENTATION for cervical spine stenosis.

안정성 최적화 + 향상된 데이터 증강 버전
- 안정성 최적화 (vit-small_evax-pretrain_stable.py 기반):
  1. Base LR: 2e-3 → 1e-3 (CRITICAL: gradient explosion 방지)
  2. Gradient clipping 추가: max_norm=1.0 (CRITICAL: ViT 안정성)
  3. Warmup 연장: 5 → 10 epochs (CRITICAL: 초기 불안정성 방지)
  4. Layer decay rate: 0.65 → 0.70 (균형잡힌 gradient flow)
  5. Drop path rate: 0.1 → 0.05 (작은 데이터셋에 맞춤)
  6. Max epochs: 100 → 150 (ViT는 더 긴 학습 필요)

- 향상된 데이터 증강 (9가지 transforms):
  1. RandomResizedCrop (80-100% crop + scale jittering)
  2. RandomFlip (horizontal 50% + vertical 30%)
  3. RandomAffine (±10% translation, 90-110% scale)
  4. Rotate (±15°, increased from ±10°)
  5. Shear (최대 10° horizontal)
  6. PhotoMetricDistortion (brightness/contrast variation)
  7. GaussianBlur (X-ray quality variation simulation)
  8. RandomErasing (2-15% area, occlusion robustness)
  9. Final resize to 224x224

Expected improvements:
- 안정적이고 부드러운 수렴 (no gradient explosion, no NaN)
- 향상된 일반화 성능 (diverse augmentation)
- Validation F1-score: 70-85% (base: 65-80%)
- ImageNet baseline 대비 3-7% 성능 향상
- 작은 의료 데이터셋에서 overfitting 감소

EVA-X Small (chest X-ray 사전학습) 계층별 학습률 감쇠 미세조정
- 의료 영상 도메인 (chest → cervical)
- 대규모 사전학습 데이터 (~520k images)
- Architecture: ViTEVA02 (RoPE + Sub-LN + SwiGLU)

Usage:
    # 1. Download EVA-X weights first
    python scripts/download_evax_weights.py --variant evax_small

    # 2. Train with stable config + enhanced augmentation
    python mmpretrain/tools/train.py \
      mmpretrain/configs/csten/finetune/vit-small_evax-pretrain_stable_enhanced-aug.py

    # 3. Monitor training (watch gradient norms + augmentation effects)
    tensorboard --logdir work_dirs/vit-small_evax-pretrain_stable_enhanced-aug/

    # 4. Compare with base augmentation version
    # - Check if train/val gap improves (better generalization)
    # - Expected: ~2-5% F1-score improvement from augmentation
"""

_base_ = [
    '../../_base_/datasets/csten_v2_enhanced_augmentation.py',  # CHANGED: Enhanced augmentation
    '../../_base_/schedules/csten_bs128_adamw.py',
    '../../_base_/csten_default_runtime.py'
]

# 커스텀 모듈 import (MultiTaskClsHeadV2 + EnhancedTensorBoardHook)
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
        patch_size=16,
        drop_path_rate=0.05,  # Reduced from 0.1 (less regularization)
        final_norm=True,
        out_type='avg_featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../checkpoints/evax/evax_small_mmpretrain.pth',
            prefix='backbone.',
        )
    ),
    neck=None,
    head=dict(
        type='MultiTaskClsHeadV2',  # V2 head (8 tasks: C3-C6 only)
        num_classes=2,  # Binary: 0=absent/mild, 1=moderate/severe
        in_channels=384,  # ViT-Small hidden dimension
        loss=dict(
            type='FocalLoss',  # Focal loss for class imbalance
            gamma=2.0,
            alpha=0.5,
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
        class_weights='auto_v2',
    ),
)

# STABILITY-OPTIMIZED OPTIMIZER CONFIGURATION
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,  # Stable LR (reduced from 2e-3)
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # Gradient clipping for stability
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.70,  # Balanced decay (0.65 → 0.70)
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
        }
    )
)

# EXTENDED WARMUP FOR STABILITY
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,  # Extended warmup (5 → 10 epochs)
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=140,  # ViT needs longer training (100 → 150 epochs)
        by_epoch=True,
        begin=10,
        end=150,
        eta_min=1e-6
    )
]

# Training configuration
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
auto_scale_lr = dict(base_batch_size=128)
work_dir = '../work_dirs/vit-small_evax-pretrain_stable_enhanced-aug'

# Visualization backends
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='c-spine-stenosis',
            name='vit-small_evax_STABLE_ENHANCED-AUG',
            tags=['finetune', 'evax', 'stable', 'enhanced-aug', 'v2-dataset'],
            group='finetuning',
            notes=(
                'EVA-X Small with STABILITY + ENHANCED AUGMENTATION. '
                'Stable config: LR 1e-3, grad clip, 10-epoch warmup, 150 epochs. '
                'Enhanced augmentation: 9 transforms including affine, blur, erasing. '
                'Expected: Stable convergence + better generalization + 3-7% improvement.'
            )
        )
    )
]

visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# Runtime hooks
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

# Note: Enhanced monitoring (gradient norms, weight histograms, etc.) is enabled by default
# via EnhancedWandbHook in csten_default_runtime.py
# No need to specify custom_hooks unless you want to override default settings

randomness = dict(seed=42, deterministic=False)

# Training notes:
# ===============
# 1. Enhanced Augmentation Benefits:
#    - 9 diverse transforms vs 3 in base config
#    - Better generalization to diverse patient positioning
#    - Robustness to X-ray quality variations and occlusions
#    - Reduced overfitting on small medical dataset
#
# 2. Expected Performance:
#    - Base config (3 augmentations): F1-score 65-80%
#    - Enhanced config (9 augmentations): F1-score 70-85%
#    - Improvement: ~2-5% from augmentation alone
#    - Combined with stability fixes: 3-7% total improvement
#
# 3. Augmentation Strategy:
#    - Geometric: Flip, rotate, affine, shear (realistic patient variation)
#    - Photometric: Brightness, contrast (X-ray quality variation)
#    - Quality: Gaussian blur (acquisition quality simulation)
#    - Regularization: Random erasing (occlusion robustness)
#
# 4. Medical Imaging Considerations:
#    - No aggressive elastic deformation (preserve anatomy)
#    - Limited rotation/shear angles (realistic positioning)
#    - Blur simulates real X-ray quality variations
#    - Random erasing mimics partial occlusions/artifacts
#
# 5. Monitoring Strategy:
#    - Watch train/val gap: Should decrease with better augmentation
#    - If train/val gap increases: Augmentation may be too strong
#    - If both train/val improve: Augmentation is helping generalization
#    - Compare with base augmentation config for A/B testing
#
# 6. Comparison Experiments:
#    Run both configs for fair comparison:
#    a. vit-small_evax-pretrain_stable.py (base augmentation)
#    b. vit-small_evax-pretrain_stable_enhanced-aug.py (this config)
#
#    Compare:
#    - Validation F1-score (primary metric)
#    - Train/val gap (generalization)
#    - Per-task performance (check if all tasks benefit)
#    - Training stability (both should be stable)
