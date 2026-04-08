# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..quantization.config import AutoModuleDecoderConfig


class ModulePreFilterProcessor(LoopProcessor):
    """Annotate modules with an ordered pre-filter plan before quantization."""

    def __init__(self, *args, **kwargs):
        """Initialize a no-forward planning processor for module pre-filters."""

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
        """Normalize configured pre-filters into a stable per-module plan."""

        del kwargs

        pipeline = []
        for pre_filter in getattr(self.qcfg, "pre_filters", []) or []:
            if isinstance(pre_filter, AutoModuleDecoderConfig):
                pipeline.append(
                    {
                        "code": pre_filter.code,
                        "source_dtype": pre_filter.source_dtype,
                        "target_dtype": pre_filter.target_dtype,
                        "forward_policy": pre_filter.forward_policy,
                        "quant_policy": pre_filter.quant_policy,
                    }
                )

        if pipeline:
            module.state["pre_filter_pipeline"] = pipeline
            module.state["auto_module_decoder"] = pipeline[-1]
        else:
            module.state.pop("pre_filter_pipeline", None)
            module.state.pop("auto_module_decoder", None)

    def is_skipped(self, module: NamedModule) -> bool:
        """Report that every candidate module passes through pre-filter planning."""

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

        return "module-pre-filter"


__all__ = ["ModulePreFilterProcessor"]
