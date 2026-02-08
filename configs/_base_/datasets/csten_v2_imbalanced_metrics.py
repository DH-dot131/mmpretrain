# dataset settings for annotations_v2 (filtered: C3-C6 only, 8 tasks)
# 경추 척추관 협착증 (Cervical Spine Stenosis) 측면 X-ray 다중 레이블 분류 - v2
dataset_type = 'MultiTaskDataset'

# 데이터 전처리 설정
# X-ray는 grayscale이므로 3채널로 복제 후 정규화
data_preprocessor = dict(
    # Grayscale X-ray용 정규화 파라미터 (ImageNet mean/std 사용)
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # Grayscale을 RGB로 변환 (3채널 복제)
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# 학습 데이터 증강 파이프라인
# 의료 영상 특성상 과도한 증강은 피하고 적절한 수준 유지
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, keep_ratio=True),
    dict(type='Pad', size=(224, 224)),
    # 수평/수직 뒤집기 (좌우 대칭성 활용)
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    # 회전 증강 (±10도 이내로 제한)
    dict(
        type='Rotate',
        angle=50,  # random_negative_prob=0.5 (default)로 ±10도 범위 커버
        prob=0.5,
        interpolation='bicubic',
        pad_val=0
    ),
    # 색상 지터 (대비 및 밝기 조정)
    dict(
        type='ColorJitter',
        brightness=0.3,
        contrast=0.3,
        saturation=0.0,  # Grayscale이므로 saturation 변경 없음
        hue=0.0,
    ),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

# 검증/테스트 파이프라인 (증강 없음)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=224, keep_ratio=True),
    dict(type='Pad', size=(224, 224)),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

# 학습 데이터 로더 설정
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations_v2/train.json',
        data_root='',  # img_path가 절대경로이므로 빈 문자열 사용
        data_prefix='',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# 검증 데이터 로더 설정
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations_v2/val.json',
        data_root='',  # img_path가 절대경로이므로 빈 문자열 사용
        data_prefix='',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 테스트 데이터 로더 설정
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations_v2/test.json',
        data_root='',  # img_path가 절대경로이므로 빈 문자열 사용
        data_prefix='',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# COMPREHENSIVE METRICS FOR IMBALANCED DATA
# Critical: Do NOT rely on accuracy alone for imbalanced datasets
# V2: Only C3-C6 tasks (8 tasks, excludes C1_C2/C2_C3/C7_T1)

task_names = [
    'C3_C4_central', 'C3_C4_foraminal',
    'C4_C5_central', 'C4_C5_foraminal',
    'C5_C6_central', 'C5_C6_foraminal',
    'C6_C7_central', 'C6_C7_foraminal'
]

task_metrics = {}
for task_name in task_names:
    task_metrics[task_name] = [
        # Basic accuracy (for reference, but can be misleading)
        dict(type='Accuracy', topk=(1,)),

        # CRITICAL METRICS FOR IMBALANCED DATA:
        # SingleLabelMetric provides precision, recall, f1-score, and support
        # 1. Precision (PPV): Of all predicted positives, how many are correct?
        # 2. Recall (Sensitivity/TPR): Of all actual positives, how many did we detect?
        # 3. F1-Score: Harmonic mean of precision and recall
        # 4. Support: Number of samples per class
        dict(
            type='SingleLabelMetric',
            items=('precision', 'recall', 'f1-score', 'support'),
            average='macro',  # Average across classes
            num_classes=2,
        ),

        # 5. Confusion Matrix: Detailed error analysis
        #    Essential for understanding FP vs FN trade-offs
        dict(
            type='ConfusionMatrix',
            num_classes=2,
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
# - Expected metrics for V2 filtered dataset:
#   * Accuracy: 75-85% (better than v1 due to less severe imbalance)
#   * F1-Score: 60-80% (target metric)
#   * Recall: 70-90% (critical - must detect stenosis)
#   * Precision: 50-75% (acceptable trade-off)