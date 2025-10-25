# optimizer
optim_wrapper = dict(
    loss_scale=512.0,
    optimizer=dict(lr=5e-4, type='AdamW', weight_decay=0.0001),
    type='AmpOptimWrapper')

# learning policy
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
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
