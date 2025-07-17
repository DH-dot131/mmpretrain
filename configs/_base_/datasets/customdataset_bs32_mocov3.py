# dataset settings
dataset_type = 'CustomDataset'
data_root = '../../raw_data/Spine_xray/BUU_LSPINE_V2/LA'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[0.5],
    std=[0.5],
    to_rgb=False)
'''
view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=1.),
    dict(type='Solarize', thr=128, prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.1),
    dict(type='Solarize', thr=128, prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]
'''
# view_pipeline 예시 (contrastive learning용)
view_pipeline = [
    dict(type='RandomResizedCrop', scale=224, crop_ratio_range=(0.2, 1.)),
    dict(type='GaussianBlur', magnitude_range=(0.1, 2.0), prob=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Rotate', angle=10, prob=0.3),
    #dict(type='RandomAffine', max_rotate_degree=5, max_translate_ratio=0.05, prob=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=[2],
        #transforms=[view_pipeline1, view_pipeline2]),
        transforms=[view_pipeline]),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file = "../file_list.txt",
        with_label=False,
        #split='train',
        pipeline=train_pipeline))
