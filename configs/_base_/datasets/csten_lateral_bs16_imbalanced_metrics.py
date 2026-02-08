# dataset settings with comprehensive metrics for imbalanced data
# 경추 척추관 협착증 (Cervical Spine Stenosis) - Imbalanced data evaluation

dataset_type = 'MultiTaskDataset'

# Data preprocessing (same as base config)
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, keep_ratio=True),
    dict(type='Pad', size=(224, 224)),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(
        type='Rotate',
        angle=50,
        prob=0.5,
        interpolation='bicubic',
        pad_val=0
    ),
    dict(
        type='ColorJitter',
        brightness=0.3,
        contrast=0.3,
        saturation=0.0,
        hue=0.0,
    ),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, keep_ratio=True),
    dict(type='Pad', size=(224, 224)),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

# Data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations/train.json',
        data_root='',
        data_prefix='',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations/val.json',
        data_root='',
        data_prefix='',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations/test.json',
        data_root='',
        data_prefix='',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# COMPREHENSIVE METRICS FOR IMBALANCED DATA
# Critical: Do NOT rely on accuracy alone for imbalanced datasets

task_names = [
    'C1_C2_central', 'C1_C2_foraminal',
    'C2_C3_central', 'C2_C3_foraminal',
    'C3_C4_central', 'C3_C4_foraminal',
    'C4_C5_central', 'C4_C5_foraminal',
    'C5_C6_central', 'C5_C6_foraminal',
    'C6_C7_central', 'C6_C7_foraminal',
    'C7_T1_central', 'C7_T1_foraminal'
]

task_metrics = {}
for task_name in task_names:
    task_metrics[task_name] = [
        # Basic accuracy (for reference, but can be misleading)
        dict(type='Accuracy', topk=(1,)),

        # CRITICAL METRICS FOR IMBALANCED DATA:

        # 1. Precision (PPV): Of all predicted positives, how many are correct?
        #    Important for avoiding false positives
        dict(
            type='Precision',
            average='macro',  # Average across classes
        ),

        # 2. Recall (Sensitivity/TPR): Of all actual positives, how many did we detect?
        #    MOST IMPORTANT for medical imaging (cannot miss stenosis)
        dict(
            type='Recall',
            average='macro',
        ),

        # 3. F1-Score: Harmonic mean of precision and recall
        #    Best single metric for imbalanced data
        dict(
            type='F1Score',
            average='macro',
        ),

        # 4. AUROC (AUC): Area under ROC curve
        #    Threshold-independent metric, good for overall discrimination
        dict(
            type='AUC',
            # For binary classification, reports single AUC value
        ),

        # 5. Confusion Matrix: Detailed error analysis
        #    Essential for understanding FP vs FN trade-offs
        dict(
            type='ConfusionMatrix',
            num_classes=2,
        ),

        # Optional: Support count (number of samples per class)
        dict(
            type='Support',
            average='macro',
        ),
    ]

# Evaluator configuration
val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics=task_metrics,
)

test_evaluator = val_evaluator

# Additional notes for interpretation:
# - Accuracy can be high even with poor minority class detection
# - Focus on F1-score, recall, and AUROC for imbalanced tasks
# - For medical applications, prioritize RECALL (sensitivity) over precision
#   Reason: Missing a stenosis (FN) is worse than a false positive (FP)
# - Expected metrics for severely imbalanced tasks:
#   * Accuracy: 85-95% (misleading - can predict all negative)
#   * F1-Score: 50-80% (target metric)
#   * Recall: 70-90% (critical - must detect stenosis)
#   * Precision: 40-70% (acceptable trade-off)
