# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Dict, Optional

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.logging import print_log

from mmpretrain.registry import HOOKS

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class EnhancedWandbHook(Hook):
    """Enhanced WandB Logging Hook for mmpretrain.

    Integrates all functionality from EnhancedTensorBoardHook with additional WandB features:
    - Gradient norms (layer-wise and global average)
    - Weight histograms (layer-wise)
    - Validation metrics (accuracy, precision, recall, f1-score, etc.)
    - Learning rate schedule
    - Model parameter statistics
    - System metrics (GPU memory, CPU usage)
    - Gradient distributions

    This hook replaces EnhancedTensorBoardHook and provides a unified logging solution.

    Args:
        log_grad_norm (bool): Whether to log gradient norms. Defaults to True.
        log_weight_hist (bool): Whether to log weight histograms. Defaults to True.
        log_val_metrics (bool): Whether to log validation metrics. Defaults to True.
        log_lr (bool): Whether to log learning rate. Defaults to True.
        log_model_stats (bool): Whether to log model statistics. Defaults to True.
        log_system_metrics (bool): Whether to log system metrics. Defaults to True.
        grad_norm_interval (int): Interval for logging gradient norms. Defaults to 50.
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
        log_system_metrics: bool = True,
        grad_norm_interval: int = 50,
        weight_hist_interval: int = 1,
        max_layers_to_log: int = 50,
    ):
        if wandb is None:
            raise ImportError(
                'wandb is not installed. Please install it with: '
                'pip install wandb'
            )

        self.log_grad_norm = log_grad_norm
        self.log_weight_hist = log_weight_hist
        self.log_val_metrics = log_val_metrics
        self.log_lr = log_lr
        self.log_model_stats = log_model_stats
        self.log_system_metrics = log_system_metrics
        self.grad_norm_interval = grad_norm_interval
        self.weight_hist_interval = weight_hist_interval
        self.max_layers_to_log = max_layers_to_log

        self._wandb_initialized = False

    def _get_model(self, runner: Runner) -> torch.nn.Module:
        """Get the actual model, unwrapping DDP/FSDP if necessary."""
        model = runner.model
        # Unwrap DDP/FSDP wrapper
        if hasattr(model, 'module'):
            return model.module
        return model

    def _ensure_wandb_initialized(self) -> bool:
        """Ensure WandB is initialized and ready."""
        if self._wandb_initialized:
            return wandb.run is not None

        # Check if wandb.run exists (initialized by WandbVisBackend)
        if wandb.run is None:
            print_log(
                'WandB run is not initialized. Make sure WandbVisBackend is added to vis_backends.',
                logger='current',
                level=logging.WARNING
            )
            return False

        self._wandb_initialized = True
        print_log(
            'EnhancedWandbHook: WandB integration active',
            logger='current',
            level=logging.INFO
        )
        return True

    @master_only
    def _log_gradient_norms(self, runner: Runner) -> None:
        """Log gradient norms for each layer and global average.

        Optimized to minimize GPU-CPU synchronization.
        """
        if not self.log_grad_norm or not self._ensure_wandb_initialized():
            return

        model = self._get_model(runner)
        if not isinstance(model, torch.nn.Module):
            return

        # Collect all gradient norms as tensors first (GPU에서 계산)
        grad_norms = []
        layer_grad_norms = {}  # For layer-wise logging
        total_params = 0
        params_with_grad = 0

        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                norm = param.grad.data.norm(2)
                grad_norms.append(norm)

                # Store for individual logging (limited)
                if len(layer_grad_norms) < self.max_layers_to_log:
                    # Simplify layer name for readability
                    layer_name = name.replace('.', '/')
                    layer_grad_norms[f'train/gradient_norm/layers/{layer_name}'] = norm.item()

        if not grad_norms:
            return

        # Calculate total norm on GPU
        grad_norms_tensor = torch.stack(grad_norms)
        total_norm = grad_norms_tensor.norm(2)
        total_norm_value = total_norm.item()

        # Log to WandB
        log_dict = {
            'train/gradient_norm/global': total_norm_value,
            'train/gradient_norm/num_params_with_grad': params_with_grad,
        }
        log_dict.update(layer_grad_norms)

        wandb.log(log_dict, step=runner.iter)

    @master_only
    def _log_weight_histograms(self, runner: Runner) -> None:
        """Log weight histograms for each layer."""
        if not self.log_weight_hist or not self._ensure_wandb_initialized():
            return

        model = self._get_model(runner)
        if not isinstance(model, torch.nn.Module):
            return

        log_dict = {}
        layer_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad and param.data.numel() > 0:
                layer_name = name.replace('.', '/')

                try:
                    param_data = param.data.detach().cpu().numpy()

                    # Sample if too large to avoid memory issues
                    if param_data.size > 100000:
                        import numpy as np
                        indices = np.random.choice(param_data.size, 100000, replace=False)
                        param_data = param_data.flatten()[indices]

                    # WandB histogram
                    log_dict[f'weights/{layer_name}'] = wandb.Histogram(param_data)

                    # Also log weight statistics
                    log_dict[f'weights_stats/{layer_name}/mean'] = float(param_data.mean())
                    log_dict[f'weights_stats/{layer_name}/std'] = float(param_data.std())

                except Exception as e:
                    print_log(
                        f'Failed to log histogram for {layer_name}: {e}',
                        logger='current',
                        level=logging.DEBUG
                    )

                layer_count += 1
                if layer_count >= self.max_layers_to_log:
                    break

        if log_dict:
            wandb.log(log_dict, step=runner.epoch)

    @master_only
    def _log_validation_metrics(self, runner: Runner,
                                metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics to WandB."""
        if not self.log_val_metrics or not self._ensure_wandb_initialized():
            return

        if metrics is None or not metrics:
            return

        log_dict = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                log_dict[f'val/{metric_name}'] = metric_value

        if log_dict:
            wandb.log(log_dict, step=runner.epoch)

    @master_only
    def _log_learning_rate(self, runner: Runner) -> None:
        """Log current learning rate."""
        if not self.log_lr or not self._ensure_wandb_initialized():
            return

        if hasattr(runner, 'optim_wrapper') and runner.optim_wrapper is not None:
            if hasattr(runner.optim_wrapper, 'param_groups'):
                log_dict = {}

                for i, param_group in enumerate(runner.optim_wrapper.param_groups):
                    lr = param_group.get('lr', None)
                    if lr is not None:
                        group_name = f'group_{i}' if len(runner.optim_wrapper.param_groups) > 1 else 'lr'
                        log_dict[f'train/learning_rate/{group_name}'] = lr

                if log_dict:
                    wandb.log(log_dict, step=runner.iter)

    @master_only
    def _log_model_statistics(self, runner: Runner) -> None:
        """Log model parameter statistics."""
        if not self.log_model_stats or not self._ensure_wandb_initialized():
            return

        model = self._get_model(runner)
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

        log_dict = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/model_size_mb': total_size / (1024 ** 2),
            'model/trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        }

        wandb.log(log_dict, step=runner.epoch)

    @master_only
    def _log_system_metrics(self, runner: Runner) -> None:
        """Log system metrics (GPU memory, etc.)."""
        if not self.log_system_metrics or not self._ensure_wandb_initialized():
            return

        try:
            if torch.cuda.is_available():
                log_dict = {}

                # GPU memory for each device
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)  # GB

                    log_dict[f'system/gpu_{i}/memory_allocated_gb'] = allocated
                    log_dict[f'system/gpu_{i}/memory_reserved_gb'] = reserved
                    log_dict[f'system/gpu_{i}/memory_max_allocated_gb'] = max_allocated

                if log_dict:
                    wandb.log(log_dict, step=runner.iter)

        except Exception as e:
            print_log(
                f'Failed to log system metrics: {e}',
                logger='current',
                level=logging.DEBUG
            )

    def before_optimizer_step(self, runner: Runner, optimizer, optimizer_idx: int = 0) -> None:
        """Called before optimizer.step(). Gradient is available here."""
        # Log gradient norms (interval-based)
        if self.log_grad_norm and runner.iter % self.grad_norm_interval == 0:
            self._log_gradient_norms(runner)

    def after_train_iter(self, runner: Runner, batch_idx: int,
                         data_batch: dict = None,
                         outputs: dict = None) -> None:
        """Called after each training iteration."""
        # Log learning rate (interval-based)
        if self.log_lr and runner.iter % self.grad_norm_interval == 0:
            self._log_learning_rate(runner)

        # Log system metrics (interval-based)
        if self.log_system_metrics and runner.iter % (self.grad_norm_interval * 2) == 0:
            self._log_system_metrics(runner)

    def after_train_epoch(self, runner: Runner) -> None:
        """Called after each training epoch."""
        # Log weight histograms (epoch-based interval)
        if self.log_weight_hist and runner.epoch % self.weight_hist_interval == 0:
            self._log_weight_histograms(runner)

        # Log model statistics
        if self.log_model_stats:
            self._log_model_statistics(runner)

    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict] = None) -> None:
        """Called after validation epoch."""
        if not self.log_val_metrics:
            return

        # Use provided metrics or get from message hub
        if metrics is None:
            try:
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
