# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict, Optional

import torch

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..nn_modules.qlinear.torch import TorchLinear
from ..utils.logger import setup_logger

log = setup_logger()

class DequantizeProcessor(LoopProcessor):
    """Restores quantized weights to dense tensors for comparison or recovery flows."""

    def __init__(self, quantized_modules: Dict[str, TorchLinear]):
        """Initializes the processor with the quantized modules to dequantize."""

        super().__init__(
            tokenizer=None,
            qcfg=None,
            calibration=None,
            calibration_concat_size=None,
            prepare_dataset_func=None,
            batch_size=1,
            execution_config=ExecutionConfig(
                require_fwd=False,
                fwd_replay_after_process=False,
            ),
        )

        self.quantized_modules = quantized_modules

    def set_calibration_dataset(self, calibration_dataset):
        """Disables calibration inputs because dequantization is weight-only."""

        self.calibration_dataset = None
        self.num_batches = 0

    # de-quantize weights
    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Dequantizes a module, preserving tensor-parallel padding when needed."""

        device = module.weight.device

        # TODO fix num_itr param..need to calculate this before dequant
        with self.lock:
            m = self.quantized_modules.pop(module.full_name)
            m.optimize()
        # log.info(f"Dequantize: `{m.name}`")

        # TODO: we can optimize this and dequant + w - wq on cpu
        tp_info = module.state.get("tp_pad_info")
        pad_cols = 0
        if isinstance(tp_info, dict):
            pad_cols = int(tp_info.get("pad_cols", 0) or 0)

        wq = m.dequantize_weight().T.to(device=device)
        if pad_cols:
            pad = wq.new_zeros((wq.shape[0], pad_cols))
            wq = torch.cat((wq, pad), dim=1)

        module_weight = module.weight.data
        if pad_cols:
            pad = module_weight.new_zeros((module_weight.shape[0], pad_cols))
            module_weight = torch.cat((module_weight, pad), dim=1)

        # diff in float32
        w_wq_diff = module_weight.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

        module.state.update({
            "w_wq_diff": w_wq_diff,
            "wq": wq,
        })

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Drops temporary dequantization tensors after downstream consumers finish."""

        module.state.pop("w", None)  # no need for these weights now
        module.state.pop("wq", None) # no need for these weights now

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Reports that no calibration dataset is required for this processor."""

        return False

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "de-quantize"
