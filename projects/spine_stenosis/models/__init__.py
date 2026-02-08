# Copyright (c) OpenMMLab. All rights reserved.
from .multi_task_head import MultiTaskClsHead
from .multi_task_head_weighted import MultiTaskClsHeadWeighted
from .multi_task_head_v2 import MultiTaskClsHeadV2

__all__ = ['MultiTaskClsHead', 'MultiTaskClsHeadWeighted', 'MultiTaskClsHeadV2']
