_base_ = [
    'resnest50-randint-adaw-lr5e-4-warm5-cos-bs32-ep100.py'
]

base_dataset = dict(
    ann_file= '../data/LSTV_classification/LAT_v3/train+val.txt',
)

test_dataloader = dict(
    dataset=dict(
        ann_file='../data/LSTV_classification/LAT_v3/test.txt',),
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=0.0001),
)

param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
    type='CosineAnnealingLR',
    by_epoch=True, 
    T_max=200,
    eta_min=1e-6)
    ]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=5)

# 모델에 dropout 추가
model = dict(
    head=dict(
        type='StackedLinearClsHead',
        num_classes=2,
        in_channels=2048,
        mid_channels=[512],
        dropout_rate=0.2,  # hidden layer 뒤 20% dropout
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=2,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
)
work_dir = '../work_dirs/lstv_classification_cropped/'