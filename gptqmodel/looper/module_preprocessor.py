# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import transformers

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..quantization.config import AutoModuleDecoderConfig, SmootherConfig, TensorParallelPadderConfig


def _get_number_of_rows_and_cols(layer: torch.nn.Module) -> tuple[int, int]:
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        return layer.weight.shape[1], layer.weight.shape[0]

    return layer.weight.shape[0], math.prod(layer.weight.shape[1:])


class ModulePreProcessor(LoopProcessor):
    """Annotate modules with an ordered preprocessor plan before quantization."""

    _TP_TARGETS = (2, 4, 8)

    def __init__(self, *args, **kwargs):
        """Initialize a no-forward planning processor for module preprocessors."""

        kwargs = dict(kwargs)
        kwargs.pop("calculate_w_wq_diff", None)
        qcfg = kwargs.pop("qcfg")
        tokenizer = kwargs.pop("tokenizer", None)
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            calibration_concat_separator=None,
            batch_size=1,
            execution_config=ExecutionConfig(
                require_fwd=False,
                fwd_replay_after_process=False,
            ),
        )
        self.qcfg = qcfg

    def preprocess(self, module: NamedModule, **kwargs):
        """Normalize configured preprocessors into a stable per-module plan."""

        del kwargs

        pipeline: List[Dict[str, Any]] = []
        auto_module_decoder_plan = None
        tp_pad_info = None
        for preprocessor in getattr(self.qcfg, "preprocessors", []) or []:
            if isinstance(preprocessor, AutoModuleDecoderConfig):
                auto_module_decoder_plan = {
                    "code": preprocessor.code,
                    "source_dtype": preprocessor.source_dtype,
                    "target_dtype": preprocessor.target_dtype,
                }
                pipeline.append(auto_module_decoder_plan)
                continue
            if isinstance(preprocessor, TensorParallelPadderConfig):
                tp_pad_info = self._compute_tp_pad_info(module)
                pipeline.append(
                    {
                        "code": preprocessor.code,
                        **tp_pad_info,
                    }
                )
                continue
            if isinstance(preprocessor, SmootherConfig):
                pipeline.append(preprocessor.to_dict())

        if pipeline:
            module.state["preprocessor_pipeline"] = pipeline
        else:
            module.state.pop("preprocessor_pipeline", None)

        if auto_module_decoder_plan is not None:
            module.state["auto_module_decoder"] = auto_module_decoder_plan
        else:
            module.state.pop("auto_module_decoder", None)
            module.state.pop("quant_source_module", None)
            module.state.pop("auto_module_decoder_forward_mode", None)
            module.state.pop("_auto_module_decoder_event_recorded", None)

        if tp_pad_info is not None and tp_pad_info["pad_cols"] > 0:
            module.state["tp_pad_info"] = tp_pad_info
        else:
            module.state.pop("tp_pad_info", None)

    def is_skipped(self, module: NamedModule) -> bool:
        """Report that every candidate module passes through preprocessor planning."""

        del module
        return False

    def pre_process_fwd_hook(self, name: str):
        """Return a no-op hook because planning does not inspect activations."""

        del name

        def _noop(module, inputs, output):
            del module, inputs, output
            return None

        return _noop

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Keep the planning stage side-effect free after preprocess."""

        del module, device, subset, previous_subset, subset_index, subset_total

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Report that this planning stage does not require calibration inputs."""

        del processor_index
        return False

    def name(self) -> str:
        """Return the processor label used in logs and lifecycle reporting."""

        return "module-preprocessor"

    def _compute_tp_pad_info(self, module: NamedModule) -> Dict[str, int]:
        """Calculate tensor-parallel padding metadata for one module."""

        target_multiple = math.lcm(*self._TP_TARGETS)
        group_size = getattr(self.qcfg, "group_size", -1)
        if group_size > 0:
            target_multiple = math.lcm(target_multiple, group_size)

        _, columns = _get_number_of_rows_and_cols(module)
        pad_cols = (target_multiple - (columns % target_multiple)) % target_multiple
        return {
            "pad_cols": pad_cols,
            "target_multiple": target_multiple,
            "original_columns": columns,
        }

__all__ = ["ModulePreProcessor"]
