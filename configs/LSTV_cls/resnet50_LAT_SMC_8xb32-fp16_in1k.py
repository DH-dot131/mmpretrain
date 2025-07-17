_base_ = [
    '../_base_/models/lstv_resnet50.py', '../_base_/datasets/lstv_LAT_SMC_bs32.py',
    '../_base_/schedules/lstv_bs256.py', '../_base_/default_runtime.py'
]



# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[8, 12, 15], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=2)
