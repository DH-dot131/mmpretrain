model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
       
    init_cfg=dict(
            type='Pretrained',
            checkpoint='RadImageNet_pytorch_ResNet50.pt',
            prefix='backbone.'),

    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',  # LinearClsHead → MultiLabelClsHead
        in_channels=2048,
        num_classes=5,  # 클래스 수 (예: 5개)
        loss=dict(
            type='FocalLoss',  # 또는 'CrossEntropyLoss'
            reduction='mean',
            gamma=2.0,      # false negative·positive 모두 집중
            alpha=0.25 ),     # 소수 클래스에 α 부여),
        thr = 0.5
        # topk 제거 (multilabel에서는 의미없음)
        # cal_acc=False 제거
    ),
    # Mixup도 multilabel에서는 복잡하므로 제거하거나 다른 augment 사용
    # train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)