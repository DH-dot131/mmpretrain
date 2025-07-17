_base_ = [
    '../_base_/models/hrnet/lstv_hrnet-w30.py', '../_base_/datasets/lstv_LAT_bs32.py',
    '../_base_/schedules/lstv_bs256_coslr.py', '../_base_/default_runtime.py'
]



# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)
auto_scale_lr = dict(base_batch_size=128)