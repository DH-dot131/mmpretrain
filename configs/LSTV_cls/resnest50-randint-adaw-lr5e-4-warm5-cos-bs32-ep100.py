_base_ = [
    '../_base_/models/lstv_resnest50.py', '../_base_/datasets/lstv_LAT_bs32.py',
    '../_base_/schedules/lstv_bs256.py', '../_base_/default_runtime.py'
]



# schedule settings
optim_wrapper = dict(
    loss_scale=512.0,
    optimizer=dict(lr=5e-4, type='AdamW', weight_decay=0.0001),
    type='AmpOptimWrapper')

param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
    type='CosineAnnealingLR',
    by_epoch=True, 
    T_max=100,
    eta_min=1e-6)
    ]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)

work_dir = '../work_dirs/lstv_classification_v2/fold_0'
auto_scale_lr = dict(base_batch_size=256)

# randomness = dict(deterministic=True, seed=42)
# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5),
)
