# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True, 
    T_max=30,
    eta_min=1e-6)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
