_base_ = [
    '../_base_/models/lstv_resnet50.py', '../_base_/datasets/lstv_AP_bs32.py',
    '../_base_/schedules/lstv_bs256.py', '../_base_/default_runtime.py'
]



# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)