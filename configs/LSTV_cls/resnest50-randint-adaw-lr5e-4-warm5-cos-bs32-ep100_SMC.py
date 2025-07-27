_base_ = [
'resnest50_LAT.py'
]


test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        data_root='',
        ann_file='../data/LSTV_classification/LAT_SMC/external_test.txt',
        #split='test',
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


work_dir = '../work_dirs/lstv_classification/external_SMC'
