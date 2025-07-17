

max_epochs = 30

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=max_epochs, by_epoch=True, begin=0, end=100)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
