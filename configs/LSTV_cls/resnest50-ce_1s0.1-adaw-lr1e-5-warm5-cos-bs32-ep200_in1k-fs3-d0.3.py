_base_ = [
    'resnest50-randint-adaw-lr5e-4-warm5-cos-bs32-ep100.py'
]
import os, inspect

# 현재 실행 중인 config 파일 경로
_config_path = inspect.getfile(inspect.currentframe())
# 파일명만 추출 (확장자 .py 제외)
_cfg_name = os.path.splitext(os.path.basename(_config_path))[0]

# work_dir에 동적으로 추가
work_dir = os.path.join(
    '..', 'work_dirs', 'lstv_classification_v2', _cfg_name
)

# train_pipeline = [
#                 dict(type='LoadImageFromFile'),
#                 dict(scale=224, type='Resize'),
#                 dict(angle=10, type='Rotate'),
#                 dict(direction='horizontal', prob=0.5, type='RandomFlip'),
#                 dict(
#                     brightness=0.2,
#                     contrast=0.2,
#                     hue=0.0,
#                     saturation=0.0,
#                     type='ColorJitter'),
#                 dict(type='PackInputs'),
#             ]

# base_dataset_train = dict(
#     pipeline= train_pipeline,
#     )

# train_dataloader = dict(
#     dataset=dict(
#         dataset = base_dataset_train,
#     )
# )

default_hooks = dict(
    checkpoint=dict(interval=10, type='CheckpointHook'),
    visualization=dict(enable=True, out_dir = work_dir + '/vis_test', interval=1, type='VisualizationHook')
    )



load_from = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'

model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNeSt',
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=load_from,
            prefix='backbone.'
        ),
        ),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=2,
        in_channels=2048,
        mid_channels=[512],
        dropout_rate=0.3,  # hidden layer 뒤 20% dropout
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=2,
            reduction='mean',
            loss_weight=1.0),),
    )
optim_wrapper = dict(
    loss_scale=512.0,
    optimizer=dict(lr=1e-5, type='AdamW', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(T_max=200, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR'),
]


train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)

test_evaluator = [
        dict(
        average=None,
        collect_device='gpu',
        num_classes=2,
        thrs=0.5,
        type='SingleLabelMetric'),
    dict(collect_device='gpu', num_classes=2, type='ConfusionMatrix'),
    dict(out_file_path= work_dir + '/results.pkl', type='DumpResults'),
]
