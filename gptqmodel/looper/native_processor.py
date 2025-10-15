# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..quantization.config import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import CPU, DEVICE_1

log = setup_logger()

NATIVE_INPUTS_STATE_KEY = "native_inp"

# v2 requires that we also need to capture/store non-quantized inputs
class NativeProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int,
                 require_fwd: bool = True):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
                         calibration_concat_size=calibration_concat_size,
                         calibration_sort=calibration_sort,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         require_fwd=require_fwd, fwd_after_process=False,
                         fwd_all_modules_in_single_pass=True)

        self.native_inp_caches = {}

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("NativeProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule):
        self.native_inp_caches[module.name] = []

    def is_skipped(self, module: NamedModule) -> bool:
        # TODO: Add skipping certain modules
        return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            # gptq is mutable.
            inp = inp[0].detach()

            if self.qcfg.v2_memory_device == "auto":
                v2_memory_device = DEVICE_1
            elif self.qcfg.v2_memory_device == "cpu":
                # slower but >= 4x vram memory reduction
                v2_memory_device = CPU
            elif isinstance(self.qcfg.v2_memory_device, str):
                v2_memory_device = torch.device(self.qcfg.v2_memory_device)
            elif isinstance(self.qcfg.v2_memory_device, torch.device):
                v2_memory_device = self.qcfg.v2_memory_device
            else:
                v2_memory_device = DEVICE_1

            self.native_inp_caches[name] += [inp.to(device=v2_memory_device)]
            del inp, out

        return tmp

    def process(self, module: NamedModule):
        module.state[NATIVE_INPUTS_STATE_KEY] = self.native_inp_caches.pop(module.name)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        module.state.pop(NATIVE_INPUTS_STATE_KEY, None)

    def finalize(self, model: BaseQModel, **kwargs):
        del self.native_inp_caches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            raise ValueError("NativeProcessor's calibration_dataset must be provided.")
        else:
            return True

    def name(self) -> str:
        return "native"
