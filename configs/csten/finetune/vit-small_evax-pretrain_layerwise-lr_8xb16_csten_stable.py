"""EVA-X Small finetuning for cervical spine stenosis - STABLE VERSION.

안정성 최적화 버전 (Stability-Optimized)
- 원본 대비 주요 변경사항:
  1. Base LR: 2e-3 → 1e-3 (CRITICAL: gradient explosion 방지)
  2. Gradient clipping 추가: max_norm=1.0 (CRITICAL: ViT 안정성)
  3. Warmup 연장: 5 → 10 epochs (CRITICAL: 초기 불안정성 방지)
  4. Layer decay rate: 0.65 → 0.70 (균형잡힌 gradient flow)
  5. Drop path rate: 0.1 → 0.05 (작은 데이터셋에 맞춤)
  6. Enhanced TensorBoard logging 추가 (gradient 모니터링)

Expected improvements:
- 안정적이고 부드러운 수렴 (no gradient explosion, no NaN)
- Validation F1-score 향상 (65-80% 예상)
- ImageNet baseline 대비 2-5% 성능 향상
- 150 epochs: ViT는 CNN보다 더 긴 학습이 필요 (self-attention convergence)

EVA-X Small (chest X-ray 사전학습) 계층별 학습률 감쇠 미세조정
- 의료 영상 도메인 (chest → cervical)
- 대규모 사전학습 데이터 (~520k images)
- Architecture: ViTEVA02 (RoPE + Sub-LN + SwiGLU)

Key architectural features (from paper):
- RoPE (Rotational Position Embedding): Better position encoding than absolute/learned
- Sub-LN (Sublayer Normalization): Improved gradient flow
- SwiGLU: Better activation than GELU/ReLU in FFN

Transfer learning strategy:
- Layer decay rate 0.70: Earlier layers (edges, textures) preserved with lower LR
- Later layers (semantic features) adapted with higher LR for cervical spine task
- Medical domain knowledge from chest X-ray → cervical spine X-ray

Usage:
    # 1. Download EVA-X weights first
    python scripts/download_evax_weights.py --variant evax_small

    # 2. Train with stable configuration
    python mmpretrain/tools/train.py \
      mmpretrain/configs/csten/finetune/vit-small_evax-pretrain_layerwise-lr_8xb16_csten_stable.py

    # 3. Monitor training (watch gradient norms)
    tensorboard --logdir work_dirs/vit-small_evax-pretrain_stable/

Stability monitoring checklist (first 20 epochs):
✓ Gradient norms: 0.1-10.0 range (>50 = warning, >100 = critical)
✓ Loss curves: Smooth decrease after warmup (no oscillation)
✓ LR warmup: 1e-7 → 1e-3 over 10 epochs
✓ Validation F1: 60-80% after 50 epochs
"""

_base_ = [
    '../../_base_/datasets/csten_v2_imbalanced_metrics.py',  # V2 dataset (C3-C6, 8 tasks)
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
        patch_size=16,  # EVA-X uses patch_size=16 unless specified otherwise
        drop_path_rate=0.05,  # CHANGED: 0.1 → 0.05 (reduced regularization for small dataset)
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

# STABILITY-OPTIMIZED OPTIMIZER CONFIGURATION
# LAYERWISE LEARNING RATE DECAY for transfer learning
# Strategy: Preserve low-level features, adapt high-level features
# - Earlier layers (near input): Lower LR → preserve medical imaging features from chest X-ray
# - Later layers (near classifier): Higher LR → adapt to cervical spine task
# - Layer decay rate 0.70: More balanced than 0.65 (reduces 111x LR gap)
#
# Example LR distribution for 12-layer ViT-Small with layer_decay_rate=0.70:
#   Layer 0 (patch_embed):  base_lr * 0.70^11 ≈ base_lr * 0.0197  (preserve low-level features)
#   Layer 6 (middle):       base_lr * 0.70^5  ≈ base_lr * 0.168
#   Layer 11 (last):        base_lr * 0.70^0  = base_lr * 1.0     (full LR, adapt)
#   Classifier head:        base_lr * 1.0                         (full LR, task-specific)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,  # CHANGED: 2e-3 → 1e-3 (CRITICAL FIX - prevent gradient explosion)
        weight_decay=0.05,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),  # ADDED (CRITICAL FIX - ViT stability)
    constructor='LearningRateDecayOptimWrapperConstructor',  # Enable layerwise LR
    paramwise_cfg=dict(
        layer_decay_rate=0.70,  # CHANGED: 0.65 → 0.70 (more balanced gradient flow)
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

# EXTENDED WARMUP FOR STABILITY
# Learning rate scheduler
# - Warmup: 10 epochs (CHANGED: 5 → 10, gradual LR increase for stable convergence)
# - Cosine annealing: 140 epochs (ViT typically needs more epochs than CNN)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,  # Start from base_lr * 1e-4
        by_epoch=True,
        begin=0,
        end=10,  # CHANGED: 5 → 10 epochs (CRITICAL FIX - prevent early instability)
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=140,  # CHANGED: 90 → 140 (ViT needs longer training)
        by_epoch=True,
        begin=10,  # CHANGED: 5 → 10
        end=150,  # CHANGED: 100 → 150 (ViT convergence)
        eta_min=1e-6  # Minimum LR at end of training
    )
]

# Training configuration
train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)  # CHANGED: 100 → 150

# Auto-scale LR based on batch size
# Formula: scaled_lr = base_lr * (actual_batch_size / base_batch_size)
auto_scale_lr = dict(base_batch_size=128)

# Work directory for this experiment
work_dir = '../work_dirs/vit-small_evax-pretrain_stable'

# Visualization backends
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='c-spine-stenosis',
            name='vit-small_evax_STABLE',
            tags=['finetune', 'evax', 'chest-pretrain', 'layerwise-lr', 'v2-dataset', 'stable'],
            group='finetuning',
            notes=(
                'EVA-X Small (chest X-ray pretrained) with STABILITY OPTIMIZATIONS. '
                'Architecture: ViTEVA02 (RoPE + Sub-LN + SwiGLU). '
                'Transfer learning: chest → cervical spine. '
                'Key fixes: LR 1e-3, gradient clipping, 10-epoch warmup, layer decay 0.70. '
                'Expected: Stable convergence + 2-5% improvement over ImageNet baseline.'
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

# Note: Enhanced monitoring (gradient norms, weight histograms, etc.) is enabled by default
# via EnhancedWandbHook in csten_default_runtime.py
# No need to specify custom_hooks unless you want to override default settings

# Reproducibility
randomness = dict(seed=42, deterministic=False)

# Training notes (for documentation)
# 1. Stability optimizations applied (vs original config):
#    a. Base LR reduced: 2e-3 → 1e-3 (prevent gradient explosion)
#    b. Gradient clipping added: max_norm=1.0 (ViT stability best practice)
#    c. Warmup extended: 5 → 10 epochs (smooth initial convergence)
#    d. Layer decay increased: 0.65 → 0.70 (more balanced gradient flow)
#    e. Drop path reduced: 0.1 → 0.05 (less regularization for small dataset)
#    f. Max epochs increased: 100 → 150 (ViT needs longer training than CNN)
#
# 2. Layerwise LR decay rate 0.70:
#    - Preserves low-level medical imaging features from chest X-ray pretraining
#    - Allows task-specific adaptation in later layers for cervical spine
#    - More uniform than 0.65 (reduces extreme LR differences)
#
# 3. Base LR 1e-3:
#    - Conservative approach for Vision Transformer finetuning
#    - Matches ViT-Base baseline configuration
#    - Literature-recommended range: 1e-3 to 5e-4 for medical imaging
#
# 4. Gradient clipping (max_norm=1.0):
#    - Standard practice for ViT stability (global gradient norm)
#    - Prevents gradient explosion in later layers with full base LR
#    - Minimal computational overhead
#
# 5. Extended warmup (10 epochs):
#    - Matches ResNet-50 baseline with layerwise LR
#    - Gradual LR increase prevents early training instability
#    - Especially important with layerwise decay (large LR differences)
#
# 6. Focal Loss with task weights:
#    - Addresses class imbalance in stenosis dataset
#    - Central stenosis weighted 2x (more clinically important)
#    - Auto-loaded class weights from annotations_v2/weights.json
#
# 7. Expected performance:
#    - Stable, smooth convergence (no NaN losses, no gradient explosion)
#    - Medical domain transfer (chest → cervical): 2-5% boost over ImageNet
#    - Validation F1-score: 65-80% for stenosis detection
#    - Large pretraining dataset (~520k) compensates for anatomical differences
#
# 8. Monitoring strategy (first 20 epochs):
#    - Gradient norms: Expect 0.1-10.0 range (>50 = warning, >100 = critical)
#    - Loss curves: Smooth decrease after warmup (no oscillation)
#    - LR warmup: Verify 1e-7 → 1e-3 over 10 epochs
#    - Per-task F1: Watch for imbalanced task performance
#
# 9. If training shows issues:
#    - Instability: Further reduce LR to 7e-4 or 5e-4
#    - Underfitting: Already set to 150 epochs (can increase to 200 if needed)
#    - Poor performance: Try layer_decay_rate = 0.75 (more uniform)
#
# 10. Why 150 epochs for ViT:
#    - Self-attention mechanisms converge slower than convolutions
#    - Literature shows ViT benefits from extended training (300+ epochs on ImageNet)
#    - Medical imaging: 150 epochs is reasonable given smaller dataset size
#    - ResNet baseline converges well at 100 epochs, ViT needs ~50% more
