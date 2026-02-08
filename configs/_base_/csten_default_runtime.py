# C-Spine Stenosis Project Default Runtime Configuration
# 경추 척추관 협착증 프로젝트 기본 런타임 설정
#
# Key differences from standard mmpretrain default_runtime:
# - No save_best option (saves all checkpoints at regular intervals)
# - Increased logging interval for multi-task training
# - WandB + TensorBoard as default visualizers
# - EnhancedWandbHook for comprehensive logging (gradient norms, weight histograms, etc.)

# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# import custom modules (for EnhancedWandbHook)
custom_imports = dict(
    imports=['projects.spine_stenosis.hooks'],
    allow_failed_imports=False
)

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 50 iterations (increased from 100 for better monitoring)
    logger=dict(type='LoggerHook', interval=50),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint configuration
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,            # Save every 5 epochs
        by_epoch=True,         # Save by epoch (not iteration)
        max_keep_ckpts=5,      # Keep last 5 checkpoints to save disk space
        save_last=True,        # Always save the last epoch checkpoint
        # NOTE: No save_best - we save all checkpoints at regular intervals
        # Rationale: For multi-task learning with imbalanced data, it's hard
        # to define a single "best" metric. We'll manually select the best
        # checkpoint based on comprehensive evaluation after training.
    ),

    # set sampler seed in distributed environment.
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

# set visualizer (WandB + TensorBoard as default)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='c-spine-stenosis',
            # name will be auto-generated from config file name
            # tags, group, notes should be specified in individual configs if needed
        )
    )
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=vis_backends
)

# Enhanced logging with WandB
# Logs gradient norms, weight histograms, validation metrics, learning rate, system metrics
custom_hooks = [
    dict(
        type='EnhancedWandbHook',
        log_grad_norm=True,           # Track gradient norms (critical for stability monitoring)
        log_weight_hist=True,         # Track weight distributions
        log_val_metrics=True,         # Track validation metrics
        log_lr=True,                  # Track learning rate schedule
        log_model_stats=True,         # Track model parameter statistics
        log_system_metrics=True,      # Track GPU memory usage
        grad_norm_interval=50,        # Log gradient norms every 50 iterations
        weight_hist_interval=1,       # Log weight histograms every epoch
    )
]

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)