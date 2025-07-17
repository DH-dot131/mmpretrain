_base_ = [
    '../_base_/models/stenosis_resnet50.py', '../_base_/datasets/stenosis_LAT.py',
    '../_base_/schedules/stenosis_bs256.py'
]



# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=512.)

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=8),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=True),
    
        # Early stopping 추가
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='multi-label/mAP',
        patience=5,  # 15 epoch 동안 개선 없으면 중단
        rule='greater'
    ),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
