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
    data_root='',
    ann_file= '../data/LSTV_classification/LAT_v2/train+val.txt',
    pipeline=train_pipeline,
)
'''
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='.',
        ann_file='../data/LSTV_classification/LAT_v2/train.txt',
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
        ann_file='../data/LSTV_classification/LAT_v2/val.txt',
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
        dataset=base_dataset,
        fold=0,
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
        dataset=base_dataset,
        fold=0,
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
        ann_file='../data/LSTV_classification/LAT_v2/test.txt',
        #split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# evaluation settings
val_evaluator = [dict(type='SingleLabelMetric', 
                     thrs = 0.5, 
                     num_classes=2, 
                     average = None,
                     collect_device = 'gpu'
                     ),
                 dict(type='ConfusionMatrix',
                     num_classes=2,
                     collect_device='gpu'
                     )]

# If you want standard test, please manually configure the test dataset

test_evaluator = val_evaluator