"""Vision Transformer (ViT-Base) for cervical spine stenosis classification.

경추 척추관 협착증 분류를 위한 ViT-Base 모델 설정.
- Backbone: ViT-Base/16 (ImageNet pretrained)
- Head: Multi-task classification head (21 tasks)
- Dataset: Lateral X-ray images
- Training: 100 epochs with AdamW optimizer

Usage:
    # 학습
    python mmpretrain/tools/train.py mmpretrain/configs/csten/vit-base-p16_8xb16_csten.py

    # 테스트
    python mmpretrain/tools/test.py mmpretrain/configs/csten/vit-base-p16_8xb16_csten.py work_dirs/vit-base-p16_8xb16_csten/best_accuracy_top1_epoch_*.pth
"""

_base_ = [
    '../_base_/datasets/csten_lateral_bs16.py',
    '../_base_/schedules/csten_bs128_adamw.py',
    '../_base_/default_runtime.py'
]

# 커스텀 모듈 import
custom_imports = dict(
    imports=['projects.spine_stenosis.models'],
    allow_failed_imports=False
)

# 모델 설정
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',  # ViT-Base
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        # ImageNet-21k pretrained weights (fine-tuned on ImageNet-1k)
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth',
            prefix='backbone',
        )
    ),
    neck=None,  # ViT는 CLS token 사용하므로 neck 불필요
    head=dict(
        type='MultiTaskClsHead',  # 커스텀 multi-task head
        num_classes=2,  # 각 태스크는 binary classification (0=absent/mild, 1=moderate/severe)
        in_channels=768,  # ViT-Base의 hidden dimension
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        task_weights=None,
    ),
)

# Optimizer 설정 조정 (ViT용)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,  # ViT는 더 큰 learning rate 사용
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        }
    )
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

work_dir = './work_dirs/vit-base-p16_8xb16_csten'
