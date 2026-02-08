# dataset settings for annotations_v2 with ENHANCED AUGMENTATION
# 경추 척추관 협착증 (Cervical Spine Stenosis) 측면 X-ray - 향상된 데이터 증강
#
# Enhanced augmentation strategy for medical imaging:
# - 기본 geometric transforms (flip, rotate, affine)
# - Blur/noise for quality variation simulation
# - Mild elastic deformation (preserve anatomical structure)
# - RandomErasing for occlusion robustness
# - Optimized for X-ray characteristics
#
# Changes from base config:
# 1. Added RandomAffine (translation + scale)
# 2. Added GaussianBlur (simulate X-ray quality variation)
# 3. Added ShearX/ShearY (mild perspective changes)
# 4. Added RandomErasing (occlusion robustness)
# 5. Increased rotation angle distribution
# 6. Added PhotoMetricDistortion for better contrast variation

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

# ENHANCED 학습 데이터 증강 파이프라인
# 의료 영상 특성 고려: 해부학적 구조 보존 + 다양성 증가
train_pipeline = [
    dict(type='LoadImageFromFile'),

    # 1. Resize with scale jittering (0.8-1.2x)
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.8, 1.0),  # 80-100% 크롭 (너무 과격하지 않게)
        interpolation='bicubic',
    ),

    # 2. 수평/수직 뒤집기 (좌우/상하 대칭성 활용)
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.3, direction='vertical'),  # 상하 뒤집기는 덜 자주

    # 3. Affine 변환 (translation + scale)
    # 약간의 위치 이동과 스케일 변화로 position robustness
    dict(
        type='RandomAffine',
        max_translate_ratio=0.1,  # 최대 10% 이동
        scaling_ratio_range=(0.9, 1.1),  # 90-110% 스케일
        max_shear_degree=0,  # Shear는 별도로 추가
        border_val=0,
        interpolation='bicubic',
        prob=0.5,
    ),

    # 4. 회전 증강 (±15도 이내로 확장)
    # 원래 ±10도에서 ±15도로 확장 (더 다양한 자세)
    dict(
        type='Rotate',
        angle=15,  # random_negative_prob=0.5로 ±15도 범위
        prob=0.6,  # 확률도 0.5→0.6으로 증가
        interpolation='bicubic',
        pad_val=0,
    ),

    # 5. Shear 변환 (X축/Y축 기울임)
    # 약간의 perspective 변화 시뮬레이션
    dict(
        type='Shear',
        magnitude_range=(0, 10),  # 최대 10도 기울임
        magnitude_level=5,  # 중간 레벨
        prob=0.3,
        direction='horizontal',  # X축 방향 shear
        pad_val=0,
        interpolation='bicubic',
    ),

    # 6. PhotoMetricDistortion (밝기/대비/채도 조정)
    # ColorJitter보다 더 강력한 photometric 변환
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,  # ±32 밝기 변화
        contrast_range=(0.7, 1.3),  # 70-130% 대비
        saturation_range=(0.9, 1.1),  # Grayscale이지만 약간만
        hue_delta=0,  # Grayscale이므로 hue 변경 없음
    ),

    # 7. Gaussian Blur (X-ray 화질 변화 시뮬레이션)
    # 실제 X-ray는 화질이 다양하므로 blur로 시뮬레이션
    dict(
        type='GaussianBlur',
        sigma_range=(0.1, 2.0),  # Sigma 0.1-2.0 범위
        prob=0.3,  # 30% 확률로 적용
    ),

    # 8. RandomErasing (Cutout 변형, occlusion robustness)
    # 작은 영역을 랜덤하게 마스킹하여 occlusion에 robust하게
    dict(
        type='RandomErasing',
        erase_prob=0.25,  # 25% 확률
        min_area_ratio=0.02,  # 최소 2% 영역
        max_area_ratio=0.15,  # 최대 15% 영역 (너무 크지 않게)
        aspect_range=(0.3, 3.0),
        mode='const',  # 상수값으로 채우기
        fill_color=0,  # 검정색 (X-ray background)
        fill_std=None,
    ),

    # 9. 최종 크기 조정 (augmentation 후 크기가 다를 수 있으므로)
    dict(type='Resize', scale=224, keep_ratio=False),  # keep_ratio=False로 정확히 224x224

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

# Enhanced Augmentation Summary:
# =============================
# 1. Geometric Transforms:
#    - RandomResizedCrop: 80-100% crop with scale jittering
#    - RandomFlip: horizontal (50%) + vertical (30%)
#    - RandomAffine: ±10% translation, 90-110% scale
#    - Rotate: ±15° (increased from ±10°)
#    - Shear: up to 10° (horizontal direction)
#
# 2. Photometric Transforms:
#    - PhotoMetricDistortion: brightness ±32, contrast 0.7-1.3x
#    - GaussianBlur: sigma 0.1-2.0 (30% prob, simulates X-ray quality)
#
# 3. Regularization:
#    - RandomErasing: 2-15% area masking (25% prob, occlusion robustness)
#
# 4. Medical Imaging Considerations:
#    - Preserve anatomical structure (no aggressive elastic deformation)
#    - Limited rotation/shear to realistic patient positioning
#    - Blur simulates varying X-ray acquisition quality
#    - RandomErasing for robustness to occlusion/artifacts
#
# 5. Expected Benefits:
#    - Improved generalization to diverse patient positioning
#    - Robustness to X-ray quality variations
#    - Better handling of partial occlusion/artifacts
#    - Reduced overfitting on small medical imaging dataset
#
# 6. Comparison with Base Config:
#    Base config: 3 transforms (flip, rotate, colorjitter)
#    Enhanced:    9 transforms with careful medical imaging tuning
#
# 7. Usage:
#    Replace base dataset config import with this file:
#    _base_ = ['../../_base_/datasets/csten_v2_enhanced_augmentation.py']
#
# 8. Monitoring:
#    - Watch for train/val gap (if gap increases, augmentation may be too strong)
#    - Expected: ~2-5% F1-score improvement from better generalization
#    - If overfitting persists, can further increase augmentation strength
