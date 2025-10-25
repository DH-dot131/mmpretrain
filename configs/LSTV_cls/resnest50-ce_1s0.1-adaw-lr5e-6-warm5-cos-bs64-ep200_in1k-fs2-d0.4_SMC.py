_base_ = [
'resnest50-ce_1s0.1-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs2-d0.4.py'
]


test_dataloader = dict(
    dataset=dict(
        data_root='',
        ann_file='../data/LSTV_classification/LAT_SMC/external_test.txt',
        ),
)