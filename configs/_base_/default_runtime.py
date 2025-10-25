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

    checkpoint=dict(
                type='CheckpointHook',
                interval=2,            # 몇 epoch마다(또는 iteration마다) 저장할지
                by_epoch=True,         # epoch 단위로 저장
                save_best='accuracy/top1',  # 'accuracy' 지표 기준으로 best 저장
                rule='greater',        # accuracy는 클수록 좋으므로 'greater' 
                max_keep_ckpts=5,      # 디스크에 남길 최다 체크포인트 수
                save_last=True        # 마지막 epoch 체크포인트는 남기지 않음
            ),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
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
# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
