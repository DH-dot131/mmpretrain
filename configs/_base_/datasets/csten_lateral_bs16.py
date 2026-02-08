# dataset settings
# 경추 척추관 협착증 (Cervical Spine Stenosis) 측면 X-ray 다중 레이블 분류
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
    # # 224x224 크기로 리사이즈 (일반적인 ViT/ResNet 입력 크기)
    # dict(
    #     type='RandomResizedCrop',
    #     scale=224,
    #     crop_ratio_range=(0.85, 1.0),  # 의료 영상이므로 crop 범위 제한
    #     backend='pillow',
    #     interpolation='bicubic'
    # ),
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
    # dict(
    #     type='ResizeEdge',
    #     scale=256,
    #     edge='short',
    #     backend='pillow',
    #     interpolation='bicubic'
    # ),
    dict(type='Resize', scale=224, keep_ratio=True),
    dict(type='Pad', size=(224, 224)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs', multi_task_fields=['gt_label']),
]

# 학습 데이터 로더 설정
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='../data/processed/annotations/train.json',
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
        ann_file='../data/processed/annotations/val.json',
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
        ann_file='../data/processed/annotations/test.json',
        data_root='',  # img_path가 절대경로이므로 빈 문자열 사용
        data_prefix='',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 평가 메트릭 설정
# 다중 태스크 분류이므로 MultiTasksMetric 사용
# 각 태스크(C1_C2_central, C1_C2_foraminal, ...)별로 Accuracy 계산
task_names = [
    'C1_C2_central', 'C1_C2_foraminal',
    'C2_C3_central', 'C2_C3_foraminal',
    'C3_C4_central', 'C3_C4_foraminal',
    'C4_C5_central', 'C4_C5_foraminal',
    'C5_C6_central', 'C5_C6_foraminal',
    'C6_C7_central', 'C6_C7_foraminal',
    'C7_T1_central', 'C7_T1_foraminal'
]

task_metrics = {
    task_name: [dict(type='Accuracy', topk=(1,))]
    for task_name in task_names
}

val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics=task_metrics,
)

test_evaluator = val_evaluator
