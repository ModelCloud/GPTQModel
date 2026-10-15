# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deprecated module alias for :mod:`gptqmodel.quantization.norm_tuning`."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from ..utils.logger import setup_logger
from .config import NormConfig
from .norm_tuning import tune_norm_params, validate_tuning_model


log = setup_logger()


__all__ = ["tune_rmsnorm_gains", "validate_tuning_model"]


def tune_rmsnorm_gains(
    model: nn.Module,
    teacher: nn.Module,
    batches: List[Dict[str, torch.Tensor]],
    module_names: List[str],
    config: NormConfig,
) -> Dict[str, Any]:
    """Deprecated alias for :func:`gptqmodel.quantization.norm_tuning.tune_norm_params`."""
    log.warn.once("`tune_rmsnorm_gains` is deprecated; use `gptqmodel.quantization.norm_tuning.tune_norm_params`.")
    return tune_norm_params(model, teacher, batches, module_names, config)
