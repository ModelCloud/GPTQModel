# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import torch

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..utils.logger import setup_logger

log = setup_logger()

class DequantizeProcessor(LoopProcessor):
    def __init__(self, quantized_modules: Dict[str, TorchQuantLinear]):
        super().__init__(tokenizer=None, qcfg=None, calibration_dataset=None, calibration_dataset_concat_size=None,
                         prepare_dataset_func=None, batch_size=1,
                         logger_board="", require_fwd=False)

        self.quantized_modules = quantized_modules

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = None
        self.num_batches = 0

    # de-quantize weights
    def process(self, module: NamedModule, auto_gc: bool = True):
        device = module.weight.device
        w = module.weight.data

        # TODO fix num_itr param..need to calculate this before dequant
        with self.lock:
            m = self.quantized_modules.pop(module.full_name)
            m.optimize()
        log.info(f"Dequantize: `{m.name}`")

        # TODO: we can optimize this and dequant + w - wq on cpu
        wq = m.dequantize_weight().T.to(device=device)

        if module.weight.data.dtype == torch.float16:
            # diff in float16
            w_wq_diff = w - wq
        else:
            # diff in float32
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

        module.state.update({
            "w_wq_diff": w_wq_diff,
            "wq": wq,
        })

    def submodule_finalize(self, module: NamedModule):
        module.state.pop("w", None)  # no need for these weights now
        module.state.pop("wq", None) # no need for these weights now

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        return False

    def name(self) -> str:
        return "de-quantize"
