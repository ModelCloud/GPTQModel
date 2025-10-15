# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Preprocessing stage that pads modules for tensor-parallel quantization."""

from __future__ import annotations

import math
from typing import Dict

import torch

from ..quantization.gptq import get_number_of_rows_and_cols
from ..utils.logger import setup_logger
from .loop_processor import LoopProcessor
from .named_module import NamedModule

log = setup_logger()


class TensorParallelWeightProcessor(LoopProcessor):
    """Annotate modules with tensor-parallel padding metadata before quantisation.

    Quantisation backends that shard weights across tensor-parallel ranks expect
    column dimensions to align with the shard size. This processor computes the
    minimal zero padding required to satisfy splits of 2, 4, and 8 and records
    the result on the wrapped module so later stages can materialise padded
    clones without mutating the live model parameters.
    """

    _TP_TARGETS = (2, 4, 8)

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("calculate_w_wq_diff", None)
        kwargs.setdefault("require_fwd", False)
        kwargs.setdefault("fwd_after_process", False)
        super().__init__(*args, **kwargs)

        self._target_multiple = math.lcm(*self._TP_TARGETS)

    def preprocess(self, module: NamedModule):  # pragma: no cover - simple hook
        # The processor operates on every eligible module; no setup required.
        pass

    def is_skipped(self, module: NamedModule) -> bool:  # pragma: no cover - always active
        return False

    def pre_process_fwd_hook(self, name: str):  # pragma: no cover - no hook data needed
        def _noop(module, inputs, output):
            return None

        return _noop

    def process(self, module: NamedModule):
        target = module.module if isinstance(module, NamedModule) else module
        weight = getattr(target, "weight", None)
        if weight is None:
            return

        pad_info = self._compute_padding(target, module)

        if pad_info["pad_cols"] == 0:
            module.state.pop("tp_pad_info", None)
            return

        module.state["tp_pad_info"] = pad_info

        log.debug(
            "Pre-padding module %s: original_cols=%d target_multiple=%d pad_cols=%d",
            getattr(module, "full_name", repr(module)),
            pad_info["original_columns"],
            pad_info["target_multiple"],
            pad_info["pad_cols"],
        )

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        # Reuse the shared calibration cache; no bespoke dataset handling needed.
        return True

    def name(self) -> str:
        return "tp-pre-pad"

    def _compute_padding(self, module: torch.nn.Module, named: NamedModule) -> Dict[str, int]:
        rows, columns = get_number_of_rows_and_cols(named)
        pad_cols = (self._target_multiple - (columns % self._target_multiple)) % self._target_multiple

        return {
            "pad_cols": pad_cols,
            "target_multiple": self._target_multiple,
            "original_columns": columns,
        }

__all__ = ["TensorParallelWeightProcessor"]
