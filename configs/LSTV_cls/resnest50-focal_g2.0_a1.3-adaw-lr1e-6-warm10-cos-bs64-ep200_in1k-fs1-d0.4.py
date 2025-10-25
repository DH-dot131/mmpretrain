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



train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        dataset=dict(
            # pipeline=[
            #     dict(type='LoadImageFromFile'),
            #     dict(scale=224, type='Resize'),
            #     dict(angle=10, type='Rotate'),
            #     dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            #     dict(
            #         brightness=0.4,
            #         contrast=0.4,
            #         hue=0.0,
            #         saturation=0.0,
            #         type='ColorJitter'),
            #     dict(type='PackInputs'),
            # ],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=224),
                dict(type='RandomFlip', direction='horizontal', prob=0.5),
                dict(type='Rotate', angle=15),
                dict(
                    brightness=0.4,
                    contrast=0.4,
                    hue=0.0,
                    saturation=0.0,
                    type='ColorJitter'),               
                dict(type='GaussianBlur', prob=0.5, magnitude_range=(0.5, 2.0)),
                dict(type='RandomErasing',
                    erase_prob=0.5,
                    min_area_ratio=0.02,
                    max_area_ratio=0.1,
                    mode = 'const',
                    fill_color = (0, 0, 0)),
                dict(type='PackInputs'),
            ],
            type='CustomDataset'),
            ),
        )

val_dataloader = dict(
    batch_size=64,
    )
test_dataloader = dict(
    batch_size=64,
    )

load_from = 'resnet50_8xb32_in1k_20210831-ea4938fc.pth'

model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNeSt',
        frozen_stages=1,
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
        dropout_rate=0.4,  # hidden layer 뒤 20% dropout
        loss=dict(
            type='FocalLoss',
            gamma=2.0,             # 보통 1~3 사이 실험
            # alpha=array_like 로 class별 가중치 지정 가능
            alpha= 0.45,
            loss_weight=1.0
        ), 
        ),
    )
optim_wrapper = dict(
    loss_scale=512.0,
    optimizer=dict(lr=1e-6, type='AdamW', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.1, type='LinearLR'),
    dict(T_max=200, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR'),
]


train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=2)

val_evaluator = [dict(type='Accuracy', thrs=0.5, collect_device='gpu'),
                dict(type='SingleLabelMetric', 
                     thrs = 0.5, 
                     num_classes=2, 
                     average = None,
                     collect_device = 'gpu'
                     ),
                 dict(type='ConfusionMatrix',
                     num_classes=2,
                     collect_device='gpu'
                     ),
                 dict(out_file_path= work_dir + '/val_results.pkl', type='DumpResults')]

test_evaluator =[dict(type='Accuracy', thrs=0.5, collect_device='gpu'),
                dict(type='SingleLabelMetric', 
                     thrs = 0.5, 
                     num_classes=2, 
                     average = None,
                     collect_device = 'gpu'
                     ),
                 dict(type='ConfusionMatrix',
                     num_classes=2,
                     collect_device='gpu'
                     ),
                 dict(out_file_path= work_dir + '/test_results.pkl', type='DumpResults')]  
