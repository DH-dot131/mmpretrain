_base_ = [
'resnest50-ce_1s0.1-adaw-lr1e-5-warm5-cos-bs32-ep200_in1k-fs3-d0.3.py'
]


test_dataloader = dict(
    dataset=dict(
        data_root='',
        ann_file='../data/LSTV_classification/LAT_SMC/external_test.txt',
        ),
)