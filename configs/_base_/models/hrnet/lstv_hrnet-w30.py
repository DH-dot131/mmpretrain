# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w30'),
    neck=[
        dict(type='HRFuseScales', in_channels=(30, 60, 120, 240)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=2,
        loss=dict(type='FocalLoss',reduction = 'mean', loss_weight=1.0),
        topk=(1, 5),
    ))
