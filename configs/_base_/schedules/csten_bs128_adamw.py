# 경추 척추관 협착증 학습 스케줄 설정
# AdamW optimizer with cosine learning rate schedule

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4,  # 초기 학습률
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,  # Normalization layer는 weight decay 적용 안함
        bias_decay_mult=0.0,  # Bias는 weight decay 적용 안함
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }
    )
)

# learning policy - Cosine annealing
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,  # 5 epoch warmup
        convert_to_iter_based=True
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=95,  # 100 - 5 (warmup)
        by_epoch=True,
        begin=5,
        end=100,
    )
]

# train, val, test setting
train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=1,  # 매 epoch마다 검증
)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# 기준 batch size: 128 (8 GPUs × 16 batch_size)
auto_scale_lr = dict(base_batch_size=128)
