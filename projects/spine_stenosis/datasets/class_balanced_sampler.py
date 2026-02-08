# Copyright (c) OpenMMLab. All rights reserved.
"""Class-balanced sampler for multi-task imbalanced data.

This sampler implements class-balanced sampling for multi-task classification,
where each task has different class distributions.

Strategy:
- Oversample minority class examples
- Undersample majority class examples (optional)
- Maintain balance across tasks

Note: Use with caution - may lead to overfitting on rare examples.
Recommended for datasets with very severe imbalance (>100:1 ratio).
"""

import math
from typing import Iterator, List, Optional, Union

import torch
from torch.utils.data import Sampler

from mmengine.dataset import BaseDataset


class MultiTaskClassBalancedSampler(Sampler):
    """Multi-task class-balanced sampler.

    This sampler balances classes within each task by:
    1. Computing sampling weights based on class frequency
    2. Sampling examples with probability inversely proportional to class frequency
    3. Optionally focusing on specific tasks with severe imbalance

    Args:
        dataset (BaseDataset): Dataset to sample from.
        samples_per_epoch (int, optional): Number of samples per epoch.
            If None, uses dataset length.
        focus_tasks (list, optional): List of task names to focus balancing on.
            If None, balances all tasks equally.
        beta (float): Temperature parameter for class-balanced sampling.
            - beta=0: Uniform sampling (no balancing)
            - beta=1: Full inverse frequency weighting
            - beta in (0,1): Soft balancing
            Default: 0.9999 (effective number of samples)
        shuffle (bool): Whether to shuffle samples. Default: True.
        seed (int): Random seed. Default: None.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        samples_per_epoch: Optional[int] = None,
        focus_tasks: Optional[List[str]] = None,
        beta: float = 0.9999,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.focus_tasks = focus_tasks
        self.beta = beta
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Compute sampling weights
        self.weights = self._compute_weights()

    def _compute_weights(self) -> torch.Tensor:
        """Compute sampling weights for class-balanced sampling.

        Returns:
            Tensor of sampling weights for each sample.
        """
        # Get metainfo from dataset
        metainfo = self.dataset.metainfo
        task_names = metainfo['tasks']

        # Count class frequencies per task
        task_class_counts = {task: {0: 0, 1: 0} for task in task_names}

        for sample in self.dataset:
            gt_label = sample['data_samples'].gt_label
            for task in task_names:
                if task in gt_label:
                    label = gt_label[task]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    task_class_counts[task][label] += 1

        # Compute effective number of samples (class-balanced loss paper)
        # E_n = (1 - beta^n) / (1 - beta)
        task_class_weights = {}
        for task in task_names:
            counts = task_class_counts[task]
            effective_num = {
                cls: (1 - self.beta ** counts[cls]) / (1 - self.beta)
                if counts[cls] > 0 else 0
                for cls in [0, 1]
            }

            # Inverse effective number = weight
            total_effective = sum(effective_num.values())
            if total_effective > 0:
                task_class_weights[task] = {
                    cls: total_effective / effective_num[cls]
                    if effective_num[cls] > 0 else 0
                    for cls in [0, 1]
                }
            else:
                task_class_weights[task] = {0: 1.0, 1: 1.0}

        # Compute per-sample weights
        # If focus_tasks is specified, only balance those tasks
        # Otherwise, average weights across all tasks
        sample_weights = []
        for sample in self.dataset:
            gt_label = sample['data_samples'].gt_label

            if self.focus_tasks:
                # Focus on specific tasks
                weights_list = []
                for task in self.focus_tasks:
                    if task in gt_label:
                        label = gt_label[task]
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        weights_list.append(task_class_weights[task][label])
                weight = sum(weights_list) / len(weights_list) if weights_list else 1.0
            else:
                # Average across all tasks
                weights_list = []
                for task in task_names:
                    if task in gt_label:
                        label = gt_label[task]
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                        weights_list.append(task_class_weights[task][label])
                weight = sum(weights_list) / len(weights_list) if weights_list else 1.0

            sample_weights.append(weight)

        return torch.tensor(sample_weights, dtype=torch.float)

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices."""
        # Set random seed for reproducibility
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        else:
            generator = None

        # Sample with replacement according to weights
        indices = torch.multinomial(
            self.weights,
            self.samples_per_epoch,
            replacement=True,
            generator=generator
        )

        if self.shuffle:
            # Shuffle indices
            if generator is not None:
                perm = torch.randperm(len(indices), generator=generator)
            else:
                perm = torch.randperm(len(indices))
            indices = indices[perm]

        return iter(indices.tolist())

    def __len__(self) -> int:
        """Number of samples per epoch."""
        return self.samples_per_epoch

    def set_epoch(self, epoch: int):
        """Set epoch for reproducibility."""
        self.epoch = epoch
