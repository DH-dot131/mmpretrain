model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeSt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone.'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(
            type='FocalLoss',
            #label_smooth_val=0.1,
            #num_classes=2,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)