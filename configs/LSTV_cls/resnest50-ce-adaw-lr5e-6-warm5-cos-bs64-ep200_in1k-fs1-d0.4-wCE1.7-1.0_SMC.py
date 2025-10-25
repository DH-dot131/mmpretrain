_base_ = [
'resnest50-ce-adaw-lr5e-6-warm5-cos-bs64-ep200_in1k-fs1-d0.4-wCE1.7-1.0.py'
]


test_dataloader = dict(
    dataset=dict(
        data_root='',
        ann_file='../data/LSTV_classification/LAT_SMC/external_test.txt',
        ),
)