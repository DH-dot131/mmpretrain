# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-task classification head for cervical spine stenosis classification.

This head handles 14 classification tasks:
- 7 cervical levels (C1-C2 through C7-T1)
- 2 stenosis types per level (central, foraminal)
- Each task is a binary classification (0: absent/mild, 1: moderate/severe)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample, MultiTaskDataSample


@MODELS.register_module()
class MultiTaskClsHead(BaseModule):
    """Multi-task classification head for spine stenosis.

    Args:
        num_classes (int): Number of classes per task. Default: 2.
        in_channels (int): Number of input feature channels.
        loss (dict): Loss function config. Default: CrossEntropyLoss.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 2048,
        loss: Dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
        task_weights: Optional[Dict[str, float]] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels

        # 태스크 목록 정의 (7 levels × 2 stenosis types)
        self.task_names = self._get_task_names()
        self.num_tasks = len(self.task_names)

        # 각 태스크별 독립적인 분류기 생성
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = nn.Linear(in_channels, num_classes)

        # Loss 함수 설정
        self.loss_module = MODELS.build(loss)

        # 태스크별 가중치 (선택적)
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

    def _get_task_names(self) -> List[str]:
        """경추 레벨과 협착 타입별 태스크 이름 생성."""
        levels = ['C1_C2', 'C2_C3', 'C3_C4', 'C4_C5', 'C5_C6', 'C6_C7', 'C7_T1']
        stenosis_types = ['central', 'foraminal']

        task_names = []
        for level in levels:
            for stenosis_type in stenosis_types:
                task_names.append(f'{level}_{stenosis_type}')

        return task_names

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """Extract features before the final classification layers.

        This method is used for feature extraction and visualization.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.

        Returns:
            Tensor: Pre-logits features of shape (batch_size, in_channels).
        """
        # Backbone의 마지막 feature map 사용
        if isinstance(feats, (tuple, list)):
            x = feats[-1]
        else:
            x = feats

        # Global average pooling이 이미 적용되어 있지 않은 경우
        if x.ndim == 4:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)

        return x

    def forward(self, feats: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.

        Returns:
            dict: Dictionary of task predictions.
                Each key is task name, value is (batch_size, num_classes) tensor.
        """
        # Get pre-logits features
        x = self.pre_logits(feats)

        # 각 태스크별 예측
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.task_heads[task_name](x)

        return predictions

    def loss(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[DataSample],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.
            data_samples (List[DataSample]): Training samples with gt_label.

        Returns:
            dict: Dictionary of losses.
        """
        # Forward pass
        predictions = self(feats)

        # Ground truth labels 추출
        gt_labels = self._extract_gt_labels(data_samples)

        # 각 태스크별 loss 계산
        losses = {}
        total_loss = 0.0

        for task_name in self.task_names:
            pred = predictions[task_name]
            target = gt_labels[task_name]

            # Loss 계산
            task_loss = self.loss_module(pred, target)

            # 태스크 가중치 적용
            weighted_loss = task_loss * self.task_weights[task_name]
            losses[f'loss_{task_name}'] = weighted_loss
            total_loss += weighted_loss

        # 전체 loss 추가
        losses['loss'] = total_loss

        return losses

    def _extract_gt_labels(
        self, data_samples: List[DataSample]
    ) -> Dict[str, torch.Tensor]:
        """Extract ground truth labels from data samples.

        Args:
            data_samples (List[DataSample]): Training samples (MultiTaskDataSample).

        Returns:
            dict: Dictionary of ground truth labels per task.
        """
        batch_size = len(data_samples)
        device = next(self.parameters()).device

        gt_labels = {
            task_name: torch.zeros(batch_size, dtype=torch.long, device=device)
            for task_name in self.task_names
        }

        for i, data_sample in enumerate(data_samples):
            # MultiTaskDataSample에서 각 태스크별 DataSample 추출
            # 각 task_name은 MultiTaskDataSample의 필드로 저장되어 있음
            for task_name in self.task_names:
                if task_name in data_sample:
                    # task_name 필드에서 DataSample 가져오기
                    task_data_sample = data_sample.get(task_name)
                    # DataSample의 gt_label 추출
                    label = task_data_sample.gt_label
                    # 텐서인 경우 스칼라로 변환, 이미 스칼라인 경우 그대로 사용
                    if isinstance(label, torch.Tensor):
                        label = label.item() if label.numel() == 1 else label
                    gt_labels[task_name][i] = label

        return gt_labels

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[DataSample]] = None,
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.
            data_samples (List[DataSample], optional): Data samples (MultiTaskDataSample).

        Returns:
            List[DataSample]: Predictions with pred_label and pred_score per task.
        """
        # Forward pass
        predictions = self(feats)

        # Softmax for probabilities
        pred_scores = {
            task_name: torch.softmax(pred, dim=1)
            for task_name, pred in predictions.items()
        }

        # Argmax for predicted class
        pred_labels = {
            task_name: torch.argmax(scores, dim=1)
            for task_name, scores in pred_scores.items()
        }

        # MultiTaskDataSample 생성 또는 업데이트
        if data_samples is None:
            batch_size = next(iter(predictions.values())).size(0)
            data_samples = [MultiTaskDataSample() for _ in range(batch_size)]

        for i, data_sample in enumerate(data_samples):
            # 각 태스크별로 DataSample 생성하고 예측 결과 저장
            for task_name in self.task_names:
                # 해당 task의 DataSample 가져오기 (없으면 새로 생성)
                has_gt = task_name in data_sample
                if has_gt:
                    task_data_sample = data_sample.get(task_name)
                else:
                    task_data_sample = DataSample()
                    data_sample.set_field(task_data_sample, task_name)

                # 예측 결과 설정
                task_data_sample.set_pred_label(pred_labels[task_name][i].item())
                task_data_sample.set_pred_score(pred_scores[task_name][i].cpu().numpy())

                # eval_mask 설정 (ground truth가 있는 경우에만 평가)
                task_data_sample.set_field(
                    has_gt,
                    'eval_mask',
                    field_type='metainfo'
                )

        return data_samples
