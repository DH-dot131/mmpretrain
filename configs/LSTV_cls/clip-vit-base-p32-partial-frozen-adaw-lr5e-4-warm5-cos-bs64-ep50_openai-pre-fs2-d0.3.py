_base_ = [
    'clip-vit-base-p32-frozen-adaw-lr1e-3-warm5-cos-bs64-ep200_openai-pre-fs2-d0.4.py'
]

# Backbone의 일부만 freeze (처음 6개 layer만 freeze)
# 뒤쪽 layer들은 fine-tuning 가능
model = dict(
    backbone=dict(
        frozen_stages=6,  # 12개 중 앞의 6개 layer만 freeze
    ),
)

# Fine-tuning하므로 learning rate를 낮춤
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0,
    optimizer=dict(
        type='AdamW',
        lr=5e-4,  # Partial fine-tuning이므로 중간 lr
        weight_decay=0.0001
    ),
)

