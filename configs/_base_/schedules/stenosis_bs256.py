# optimizer
#optim_wrapper = dict(
    #clip_grad=dict(max_norm=1.0), 
    #loss_scale='dynamic',
    #optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    optimizer = dict(type='AdamW', lr=1e-6, weight_decay=1e-5))


# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),  # Warmup
    dict(type='MultiStepLR', by_epoch=True, milestones=[30, 50, 70], gamma=0.1), 
    ]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval = 8)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
