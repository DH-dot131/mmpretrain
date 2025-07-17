# dataset settings
dataset_type = 'MultiLabelDataset'
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0.485, 0.456, 0.406],        # ImageNet mean
    std=[0.229, 0.224, 0.225], 
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', scale=224),
    #dict(type='ResizeEdge', scale=256, edge='short'),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=224),
    dict(type='Rotate', angle = 10),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
    type='ColorJitter',
    brightness=0.2,   # 밝기 조정: [0.8, 1.2] 범위에서 랜덤 스케일링
    contrast=0.2,     # 대비 조정: [0.8, 1.2]
    saturation=0.2,   # 채도 조정: [0.8, 1.2]
    hue=0.05,         # 색상(hue) 조정: [-0.05, +0.05]
    backend='pillow'
),

    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='ResizeEdge', scale=256, edge='short'),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=224),
    dict(type='PackInputs'),
]

base_dataset = dict(
    type=dataset_type,
    data_root='../../raw_data/Spine_xray/Stenosis/spinal_stenosis_v2_png',
    ann_file='Stenosis_classification/train+val_annotation.json',
    pipeline=train_pipeline,
)
'''
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.2,  # 20% 미만인 클래스는 오버샘플링
            lazy_init=True,
            dataset=dict(
                type='KFoldDataset',
                dataset=base_dataset,
                fold=0,            # 0~K-1 중 선택
                num_splits=5,      # K=5
                test_mode=False,   # train fold
                seed=42
            )
        ),
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='KFoldDataset',
        dataset=base_dataset,
        fold=0,
        num_splits=5,
        test_mode=True,    # val fold
        seed=42
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
'''
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.2,  # 20% 미만인 클래스는 오버샘플링
            lazy_init=True,
            dataset= dict(
                type=dataset_type,
                data_root='../../raw_data/Spine_xray/Stenosis/spinal_stenosis_v2_png',
                ann_file='Stenosis_classification/train_annotation.json',
                pipeline=train_pipeline,
                )
        ),
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)
'''
train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset= dict(
                type=dataset_type,
                data_root='../../raw_data/Spine_xray/Stenosis/spinal_stenosis_v2_png',
                ann_file='Stenosis_classification/train_annotation.json',
                pipeline=train_pipeline,
            ),
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)
'''
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset= dict(
                type=dataset_type,
                data_root='../../raw_data/Spine_xray/Stenosis/spinal_stenosis_v2_png',
                ann_file='Stenosis_classification/val_annotation.json',
                pipeline=test_pipeline,
            ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='../../raw_data/Spine_xray/Stenosis/spinal_stenosis_v2_png',
        ann_file='Stenosis_classification/test_annotation.json',
        #split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    #test_mode=True,
)

# evaluation settings
val_evaluator = [
                # 1) mAP 계산
                dict(
                    type='AveragePrecision',
                    collect_device='gpu'
                ),
                # 2) 클래스별 개별 지표 반환 (precision, recall, f1-score)
                dict(
                    type='MultiLabelMetric',
                    thr=0.5,                # score ≥ 0.5를 positive로 간주
                    items=('precision','recall','f1-score'),
                    average=None,           # None: 클래스별 결과, 'macro'/'micro'도 선택 가능
                    collect_device='gpu'
                ),
]

# If you want standard test, please manually configure the test dataset

test_evaluator = val_evaluator