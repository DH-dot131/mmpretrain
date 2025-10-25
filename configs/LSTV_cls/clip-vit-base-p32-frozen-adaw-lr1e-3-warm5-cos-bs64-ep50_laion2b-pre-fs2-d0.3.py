_base_ = [
    'clip-vit-base-p32-frozen-adaw-lr1e-3-warm5-cos-bs64-ep200_openai-pre-fs2-d0.4.py'
]

# LAION2B pretrained weight 사용 (OpenAI 대신)
# LAION2B는 더 큰 데이터셋으로 학습되어 일반적으로 더 좋은 성능을 보입니다
# LAION2B도 동일한 CLIP normalization 사용
load_from = 'https://download.openmmlab.com/mmclassification/v0/clip/clip-vit-base-p32_laion2b-pre_3rdparty_in1k_20221220-194df57f.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=load_from,
            prefix='backbone.'
        ),
    ),
)

