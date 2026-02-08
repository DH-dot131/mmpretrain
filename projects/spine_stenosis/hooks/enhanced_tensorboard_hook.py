# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import Dict, Optional, Sequence

import mmengine
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmengine.logging import print_log

from mmpretrain.registry import HOOKS

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


@HOOKS.register_module()
class EnhancedTensorBoardHook(Hook):
    """Enhanced TensorBoard Logging Hook for mmpretrain.

    This hook logs comprehensive training information to TensorBoard including:
    - Gradient norms (layer-wise and global average)
    - Weight histograms (layer-wise)
    - Validation metrics (accuracy, precision, recall, f1-score, etc.)
    - Learning rate schedule
    - Model parameter statistics

    Args:
        log_grad_norm (bool): Whether to log gradient norms. Defaults to True.
        log_weight_hist (bool): Whether to log weight histograms. Defaults to True.
        log_val_metrics (bool): Whether to log validation metrics. Defaults to True.
        log_lr (bool): Whether to log learning rate. Defaults to True.
        log_model_stats (bool): Whether to log model statistics. Defaults to True.
        grad_norm_interval (int): Interval for logging gradient norms. Defaults to 100.
        weight_hist_interval (int): Interval (in epochs) for logging weight histograms.
            Defaults to 1.
        max_layers_to_log (int): Maximum number of layers to log individually.
            Defaults to 50.
    """

    # Set priority to run after other hooks
    priority = 'VERY_LOW'

    def __init__(
        self,
        log_grad_norm: bool = True,
        log_weight_hist: bool = True,
        log_val_metrics: bool = True,
        log_lr: bool = True,
        log_model_stats: bool = True,
        grad_norm_interval: int = 100,
        weight_hist_interval: int = 1,
        max_layers_to_log: int = 50,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.log_grad_norm = log_grad_norm
        self.log_weight_hist = log_weight_hist
        self.log_val_metrics = log_val_metrics
        self.log_lr = log_lr
        self.log_model_stats = log_model_stats
        self.grad_norm_interval = grad_norm_interval
        self.weight_hist_interval = weight_hist_interval
        self.max_layers_to_log = max_layers_to_log

        # TensorBoard writer will be initialized lazily
        self._tb_writer = None
        self._tb_writer_initialized = False

    def _get_model(self, runner: Runner) -> torch.nn.Module:
        """Get the actual model, unwrapping DDP/FSDP if necessary.

        Args:
            runner: The runner instance.

        Returns:
            The unwrapped model.
        """
        model = runner.model
        # Unwrap DDP/FSDP wrapper
        if hasattr(model, 'module'):
            return model.module
        return model

    def _get_tensorboard_writer(self, runner: Runner):
        """Get or initialize TensorBoard writer."""
        if self._tb_writer_initialized:
            return self._tb_writer

        if SummaryWriter is None:
            print_log(
                'TensorBoard is not installed. Please install it with: '
                'pip install tensorboard',
                logger='current',
                level=logging.WARNING
            )
            self._tb_writer_initialized = True
            return None

        # Try to get TensorBoard writer from vis_backends
        for backend_name, backend in self._visualizer._vis_backends.items():
            if 'tensorboard' in backend_name.lower() or 'TensorboardVisBackend' in str(type(backend)):
                # Try multiple possible attribute names for writer
                writer_attrs = ['writer', '_writer', '_tb_writer', 'tb_writer']
                for attr in writer_attrs:
                    if hasattr(backend, attr):
                        writer = getattr(backend, attr)
                        if writer is not None:
                            self._tb_writer = writer
                            self._tb_writer_initialized = True
                            print_log(
                                f'EnhancedTensorBoardHook: Found TensorBoard writer from {backend_name} (attr: {attr})',
                                logger='current',
                                level=logging.INFO
                            )
                            return self._tb_writer

                # Try to get save_dir from backend and create writer at same location
                if hasattr(backend, 'save_dir'):
                    save_dir = backend.save_dir
                elif hasattr(backend, '_save_dir'):
                    save_dir = backend._save_dir
                else:
                    # Default: use vis_data directory (TensorboardVisBackend default)
                    if hasattr(runner, 'work_dir') and hasattr(runner, 'timestamp'):
                        save_dir = os.path.join(runner.work_dir, runner.timestamp, 'vis_data')
                    else:
                        save_dir = None

                if save_dir:
                    try:
                        self._tb_writer = SummaryWriter(log_dir=save_dir)
                        self._tb_writer_initialized = True
                        print_log(
                            f'EnhancedTensorBoardHook: Created TensorBoard writer at {save_dir} (same as TensorboardVisBackend)',
                            logger='current',
                            level=logging.INFO
                        )
                        return self._tb_writer
                    except Exception as e:
                        print_log(
                            f'Failed to create TensorBoard writer at {save_dir}: {e}',
                            logger='current',
                            level=logging.DEBUG
                        )

        # If not found, try to create one in vis_data directory (same as TensorboardVisBackend)
        if hasattr(runner, 'work_dir') and hasattr(runner, 'timestamp'):
            try:
                # Use vis_data directory to match TensorboardVisBackend
                log_dir = os.path.join(runner.work_dir, runner.timestamp, 'vis_data')
                os.makedirs(log_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                self._tb_writer_initialized = True
                print_log(
                    f'EnhancedTensorBoardHook: Created TensorBoard writer at {log_dir}',
                    logger='current',
                    level=logging.INFO
                )
                return self._tb_writer
            except Exception as e:
                print_log(
                    f'Failed to create TensorBoard writer: {e}',
                    logger='current',
                    level=logging.WARNING
                )

        self._tb_writer_initialized = True
        print_log(
            'EnhancedTensorBoardHook: No TensorBoard writer found. '
            'Please add TensorboardVisBackend to your config.',
            logger='current',
            level=logging.WARNING
        )
        return None

    @master_only
    def _log_gradient_norms(self, runner: Runner) -> None:
        """Log gradient norms for each layer and global average.

        Optimized to minimize GPU-CPU synchronization.
        """
        if not self.log_grad_norm:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            print_log(
                f'Gradient norm logging: model is not a torch.nn.Module',
                logger='current',
                level=logging.DEBUG
            )
            return

        # Collect all gradient norms as tensors first (GPU에서 계산)
        grad_norms = []
        param_info = []  # (name, norm_tensor) pairs
        total_params = 0
        params_with_grad = 0

        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                norm = param.grad.data.norm(2)
                grad_norms.append(norm)

                # Store for individual logging (limited)
                if len(param_info) < self.max_layers_to_log:
                    # Simplify layer name for readability
                    layer_name = name.replace('.', '/')
                    param_info.append((layer_name, norm))

        # Debug logging
        if total_params > 0 and params_with_grad == 0:
            print_log(
                f'Gradient norm logging: No gradients found! '
                f'Total params: {total_params}, Params with grad: {params_with_grad}',
                logger='current',
                level=logging.WARNING
            )
            return

        if not grad_norms:
            print_log(
                f'Gradient norm logging: grad_norms list is empty',
                logger='current',
                level=logging.DEBUG
            )
            return

        # Calculate total norm on GPU
        grad_norms_tensor = torch.stack(grad_norms)
        total_norm = grad_norms_tensor.norm(2)

        # Single GPU->CPU transfer for total norm
        total_norm_value = total_norm.item()

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            print_log(
                f'Gradient norm logging: TensorBoard writer is None',
                logger='current',
                level=logging.WARNING
            )
            return

        # Log global average gradient norm
        tb_writer.add_scalar(
            'train/gradient_norm/global',
            total_norm_value,
            runner.iter
        )

        # Log individual layer gradient norms (한 번에 CPU로 이동)
        for layer_name, norm_tensor in param_info:
            tb_writer.add_scalar(
                f'train/gradient_norm/layers/{layer_name}',
                norm_tensor.item(),  # 개별 변환
                runner.iter
            )

        # Flush to ensure immediate write to disk
        try:
            tb_writer.flush()
        except Exception as e:
            print_log(
                f'Failed to flush TensorBoard writer: {e}',
                logger='current',
                level=logging.WARNING
            )

        print_log(
            f'Gradient norm logged: global={total_norm_value:.4f}, '
            f'layers={len(param_info)}, iter={runner.iter}',
            logger='current',
            level=logging.DEBUG
        )

    @master_only
    def _log_weight_histograms(self, runner: Runner) -> None:
        """Log weight histograms for each layer."""
        if not self.log_weight_hist:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            return

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        layer_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.numel() > 0:
                # Simplify layer name for readability
                layer_name = name.replace('.', '/')

                # Log weight histogram - detach and move to CPU
                try:
                    param_data = param.data.detach().cpu()

                    # Sample if too large to avoid memory issues
                    if param_data.numel() > 100000:
                        indices = torch.randperm(param_data.numel())[:100000]
                        param_data = param_data.flatten()[indices]

                    tb_writer.add_histogram(
                        f'weights/{layer_name}',
                        param_data,
                        runner.epoch
                    )
                except Exception as e:
                    print_log(
                        f'Failed to log histogram for {layer_name}: {e}',
                        logger='current',
                        level=logging.DEBUG
                    )

                layer_count += 1
                if layer_count >= self.max_layers_to_log:
                    break

        # Flush to ensure immediate write to disk
        if tb_writer is not None:
            try:
                tb_writer.flush()
            except Exception as e:
                print_log(
                    f'Failed to flush TensorBoard writer: {e}',
                    logger='current',
                    level=logging.WARNING
                )

    @master_only
    def _log_validation_metrics(self, runner: Runner,
                                metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics to TensorBoard."""
        if not self.log_val_metrics:
            return

        if metrics is None or not metrics:
            return

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                # Log validation metrics
                tb_writer.add_scalar(
                    f'val/{metric_name}',
                    metric_value,
                    runner.epoch
                )

        # Flush to ensure immediate write to disk
        try:
            tb_writer.flush()
        except Exception as e:
            print_log(
                f'Failed to flush TensorBoard writer: {e}',
                logger='current',
                level=logging.WARNING
            )


    @master_only
    def _log_learning_rate(self, runner: Runner) -> None:
        """Log current learning rate."""
        if not self.log_lr:
            return

        if hasattr(runner, 'optim_wrapper') and runner.optim_wrapper is not None:
            if hasattr(runner.optim_wrapper, 'param_groups'):
                # Get TensorBoard writer
                tb_writer = self._get_tensorboard_writer(runner)
                if tb_writer is None:
                    return

                for i, param_group in enumerate(runner.optim_wrapper.param_groups):
                    lr = param_group.get('lr', None)
                    if lr is not None:
                        group_name = f'group_{i}' if len(runner.optim_wrapper.param_groups) > 1 else 'lr'
                        tb_writer.add_scalar(
                            f'train/learning_rate/{group_name}',
                            lr,
                            runner.iter)

                # Flush to ensure immediate write to disk
                try:
                    tb_writer.flush()
                except Exception as e:
                    print_log(
                        f'Failed to flush TensorBoard writer: {e}',
                        logger='current',
                        level=logging.WARNING
                    )

    @master_only
    def _log_model_statistics(self, runner: Runner) -> None:
        """Log model parameter statistics."""
        if not self.log_model_stats:
            return

        model = self._get_model(runner)  # DDP-safe
        if not isinstance(model, torch.nn.Module):
            return

        total_params = 0
        trainable_params = 0
        total_size = 0

        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            total_size += num_params * param.element_size()
            if param.requires_grad:
                trainable_params += num_params

        # Get TensorBoard writer
        tb_writer = self._get_tensorboard_writer(runner)
        if tb_writer is None:
            return

        # Log parameter counts
        tb_writer.add_scalar(
            'model/total_parameters',
            total_params,
            runner.epoch)
        tb_writer.add_scalar(
            'model/trainable_parameters',
            trainable_params,
            runner.epoch)
        tb_writer.add_scalar(
            'model/model_size_mb',
            total_size / (1024 ** 2),  # Convert to MB
            runner.epoch)

        # Flush to ensure immediate write to disk
        try:
            tb_writer.flush()
        except Exception as e:
            print_log(
                f'Failed to flush TensorBoard writer: {e}',
                logger='current',
                level=logging.WARNING
            )

    def before_optimizer_step(self, runner: Runner, optimizer, optimizer_idx: int = 0) -> None:
        """Called before optimizer.step(). Gradient is available here."""
        # Log gradient norms (interval-based) - gradient is available before step
        if self.log_grad_norm and runner.iter % self.grad_norm_interval == 0:
            self._log_gradient_norms(runner)

    def after_train_iter(self, runner: Runner, batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """Called after each training iteration."""
        # Log learning rate (interval-based)
        if self.log_lr and runner.iter % self.grad_norm_interval == 0:
            self._log_learning_rate(runner)

    def after_train_epoch(self, runner: Runner) -> None:
        """Called after each training epoch."""
        # Log weight histograms (epoch-based interval)
        if self.log_weight_hist and runner.epoch % self.weight_hist_interval == 0:
            self._log_weight_histograms(runner)

        # Log model statistics
        if self.log_model_stats:
            self._log_model_statistics(runner)

    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict] = None) -> None:
        """Called after validation epoch.

        Args:
            runner: The runner instance.
            metrics: Validation metrics dictionary. If None, will try to retrieve
                from message_hub.
        """
        if not self.log_val_metrics:
            return

        # Use provided metrics or get from message hub
        if metrics is None:
            try:
                # Try to get metrics from message hub (MMEngine 1.x 방식)
                if hasattr(runner, 'message_hub'):
                    metrics = runner.message_hub.get_scalar('val').data
            except Exception as e:
                print_log(
                    f'Failed to retrieve validation metrics from message_hub: {e}',
                    logger='current',
                    level=logging.DEBUG
                )
                return

        # Log metrics if available
        if metrics:
            self._log_validation_metrics(runner, metrics)
