_base_ = [
    'resnest50-randint-adaw-lr5e-4-warm5-cos-bs32-ep100.py'
]
import os, inspect

# 현재 실행 중인 config 파일 경로
_config_path = inspect.getfile(inspect.currentframe())
# 파일명만 추출 (확장자 .py 제외)
_cfg_name = os.path.splitext(os.path.basename(_config_path))[0]

_fold = 0
# work_dir에 동적으로 추가
work_dir = os.path.join(
    '..', 'work_dirs', 'lstv_classification_v2', _cfg_name, 'fold_' + str(_fold)
)
# 고정 경로 사용
# work_dir = '../work_dirs/lstv_classification_v2/resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4'
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
    visualization=dict(enable=True, out_dir = work_dir + '/vis_val', interval=1, type='VisualizationHook')
    )

train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        dataset=dict(
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(scale=224, type='Resize'),
                dict(angle=10, type='Rotate'),
                dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                dict(
                    brightness=0.4,
                    contrast=0.4,
                    hue=0.0,
                    saturation=0.0,
                    type='ColorJitter'),
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
        frozen_stages=2,
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
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=2,
            reduction='mean',
            loss_weight=1.0),),
    )
optim_wrapper = dict(
    loss_scale=512.0,
    optimizer=dict(lr=5e-6, type='AdamW', weight_decay=0.0001),
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


# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[0.485, 0.456, 0.406],        # ImageNet mean
    std=[0.229, 0.224, 0.225], 
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', scale=224),
    dict(type='Resize', scale=224),
    dict(type='Rotate', angle = 10),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(
        type='ColorJitter',
        brightness=0.2,   # 밝기 변화 범위 (0~1)
        contrast=0.2,     # 대비 변화 범위 (0~1)
        saturation=0.0,   # 흑백이므로 0
        hue=0.0           # 흑백이므로 0
    ),
    #     dict(
    #     type='RandomErasing',
    #     erase_prob=0.3,   # 30% 확률로 일부 영역을 지움
    #     mode='rand',
    #     min_area_ratio=0.02,
    #     max_area_ratio=0.2
    # ),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='ResizeEdge', scale=256, edge='short'),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=224),
    dict(type='PackInputs'),
]
base_dataset_train = dict(
    type=dataset_type,
    data_root='',
    ann_file= '../data/LSTV_classification/LAT_v2/train+val_backup.txt',
    pipeline=train_pipeline,
)

base_dataset_val = dict(
    type=dataset_type,
    data_root='',
    ann_file= '../data/LSTV_classification/LAT_v2/train+val_backup.txt',
    pipeline=test_pipeline,
)
'''
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='.',
        ann_file='../data/LSTV_classification/LAT_v2/train_backup.txt',
        #split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='.',
        ann_file='../data/LSTV_classification/LAT_v2/val_backup.txt',
        #split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
'''
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='KFoldDataset',
        dataset=base_dataset_train,
        fold=_fold,
        num_splits=5,
        test_mode=False,    # val fold
        seed=42
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='KFoldDataset',
        dataset=base_dataset_val,
        fold=_fold,
        num_splits=5,
        test_mode=True,    # val fold
        seed=42
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='../data/LSTV_classification/LAT_v2/test_backup.txt',
        #split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# evaluation settings
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
                     )
                 ]

# If you want standard test, please manually configure the test dataset

test_evaluator = val_evaluator