# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-task classification head with per-task class weighting for imbalanced data.

Enhanced version of MultiTaskClsHead that supports:
- Per-task class weights (loaded from JSON file or dict)
- Task exclusion (useful for tasks with zero positive samples)
- Flexible loss functions (CrossEntropyLoss, FocalLoss, etc.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample, MultiTaskDataSample


@MODELS.register_module()
class MultiTaskClsHeadWeighted(BaseModule):
    """Multi-task classification head with per-task class weighting.

    Args:
        num_classes (int): Number of classes per task. Default: 2.
        in_channels (int): Number of input feature channels.
        loss (dict): Loss function config. Default: CrossEntropyLoss.
        task_weights (dict, optional): Per-task loss weights. Tasks with weight=0 are excluded.
        class_weights (dict or str, optional): Per-task class weights.
            - If dict: {task_name: [weight_class_0, weight_class_1], ...}
            - If str: Path to JSON file containing class weights
            - If 'auto': Load from data/processed/annotations/class_weights.json
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 2048,
        loss: Dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
        task_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Union[Dict, str]] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Task list
        self.task_names = self._get_task_names()
        self.num_tasks = len(self.task_names)

        # Task-level weights (for weighting loss across tasks)
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Identify active tasks (exclude tasks with weight=0)
        self.active_tasks = [
            name for name in self.task_names
            if self.task_weights.get(name, 1.0) > 0.0
        ]

        # Create task-specific classification heads
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = nn.Linear(in_channels, num_classes)

        # Load class weights
        self.class_weights_dict = self._load_class_weights(class_weights)

        # Create loss modules per task (with task-specific class weights)
        self.loss_modules = nn.ModuleDict()
        for task_name in self.task_names:
            loss_cfg = loss.copy()

            # Add class weights if available
            if self.class_weights_dict and task_name in self.class_weights_dict:
                task_class_weights = self.class_weights_dict[task_name]
                # Convert to tensor
                weight_tensor = torch.tensor(
                    task_class_weights,
                    dtype=torch.float32
                )
                loss_cfg['weight'] = weight_tensor

            self.loss_modules[task_name] = MODELS.build(loss_cfg)

    def _get_task_names(self) -> List[str]:
        """Generate task names for all cervical levels and stenosis types."""
        levels = ['C1_C2', 'C2_C3', 'C3_C4', 'C4_C5', 'C5_C6', 'C6_C7', 'C7_T1']
        stenosis_types = ['central', 'foraminal']

        task_names = []
        for level in levels:
            for stenosis_type in stenosis_types:
                task_names.append(f'{level}_{stenosis_type}')

        return task_names

    def _load_class_weights(
        self,
        class_weights: Optional[Union[Dict, str]]
    ) -> Optional[Dict[str, List[float]]]:
        """Load per-task class weights.

        Args:
            class_weights: Class weight specification
                - None: No class weights
                - Dict: {task_name: [weight_0, weight_1], ...}
                - 'auto': Load from default location
                - str (path): Load from JSON file

        Returns:
            Dictionary of class weights per task, or None
        """
        if class_weights is None:
            return None

        if isinstance(class_weights, dict):
            return class_weights

        # Load from file
        if isinstance(class_weights, str):
            if class_weights == 'auto':
                # Default location
                weights_path = Path(__file__).parents[4] / \
                    'data' / 'processed' / 'annotations' / 'class_weights.json'
            else:
                weights_path = Path(class_weights)

            if not weights_path.exists():
                print(f"Warning: Class weights file not found: {weights_path}")
                return None

            with open(weights_path, 'r', encoding='utf-8') as f:
                weights_dict = json.load(f)

            print(f"Loaded class weights from: {weights_path}")
            return weights_dict

        return None

    def forward(self, feats: Tuple[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.

        Returns:
            dict: Dictionary of task predictions.
        """
        # Extract feature vector
        if isinstance(feats, (tuple, list)):
            x = feats[-1]
        else:
            x = feats

        # Global average pooling if not already applied
        if x.ndim == 4:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)

        # Per-task predictions
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
        """Calculate losses with per-task class weighting.

        Args:
            feats (tuple[Tensor]): Feature maps from backbone.
            data_samples (List[DataSample]): Training samples with gt_label.

        Returns:
            dict: Dictionary of losses.
        """
        # Forward pass
        predictions = self(feats)

        # Extract ground truth labels
        gt_labels = self._extract_gt_labels(data_samples)

        # Calculate per-task losses
        losses = {}
        total_loss = 0.0
        num_active_tasks = 0

        for task_name in self.task_names:
            # Skip excluded tasks
            if self.task_weights.get(task_name, 1.0) == 0.0:
                continue

            pred = predictions[task_name]
            target = gt_labels[task_name]

            # Calculate loss using task-specific loss module (with class weights)
            task_loss = self.loss_modules[task_name](pred, target)

            # Apply task-level weight
            weighted_loss = task_loss * self.task_weights[task_name]
            losses[f'loss_{task_name}'] = weighted_loss
            total_loss += weighted_loss
            num_active_tasks += 1

        # Average loss across active tasks
        if num_active_tasks > 0:
            losses['loss'] = total_loss / num_active_tasks
        else:
            losses['loss'] = total_loss

        return losses

    def _extract_gt_labels(
        self, data_samples: List[DataSample]
    ) -> Dict[str, torch.Tensor]:
        """Extract ground truth labels from data samples.

        Args:
            data_samples (List[DataSample]): Training samples.

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
            for task_name in self.task_names:
                if task_name in data_sample:
                    task_data_sample = data_sample.get(task_name)
                    label = task_data_sample.gt_label

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
            data_samples (List[DataSample], optional): Data samples.

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

        # Create or update MultiTaskDataSample
        if data_samples is None:
            batch_size = next(iter(predictions.values())).size(0)
            data_samples = [MultiTaskDataSample() for _ in range(batch_size)]

        for i, data_sample in enumerate(data_samples):
            for task_name in self.task_names:
                # Get or create task-specific DataSample
                has_gt = task_name in data_sample
                if has_gt:
                    task_data_sample = data_sample.get(task_name)
                else:
                    task_data_sample = DataSample()
                    data_sample.set_field(task_data_sample, task_name)

                # Set predictions
                task_data_sample.set_pred_label(pred_labels[task_name][i].item())
                task_data_sample.set_pred_score(pred_scores[task_name][i].cpu().numpy())

                # Set eval_mask (only evaluate active tasks with ground truth)
                is_active = self.task_weights.get(task_name, 1.0) > 0.0
                task_data_sample.set_field(
                    has_gt and is_active,
                    'eval_mask',
                    field_type='metainfo'
                )

        return data_samples
