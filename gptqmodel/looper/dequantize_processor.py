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

from typing import Optional, Dict

import torch
from gptqmodel import QuantizeConfig
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.logger import setup_logger

logger = setup_logger()

class DequantizeProcessor(LoopProcessor):
    def __init__(self, quantized_modules: Dict[str, TorchQuantLinear], tokenizer, qcfg: QuantizeConfig, calibration_dataset,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True,

                 ):
        super().__init__(tokenizer, qcfg, calibration_dataset, calibration_dataset_concat_size, batch_size,
                         logger_board, require_fwd)

        self.quantized_modules = quantized_modules

    # de-quantize weights
    def process(self, module: NamedModule):
        w = module.weight.data.to(device=CPU, dtype=torch.float16) # TODO: allow w to be native bf16 and upcast to fp32?

        # TODO fix num_itr param..need to calculate this before dequant
        wq = self.quantized_modules.pop(module.full_name).dequantize_weight(num_itr=1).to(device=CPU, dtype=torch.float16)

        module.state.update({
            "w": w,
            "wq": wq,
        })

    def submodule_finalize(self, module: NamedModule):
        module.state.pop("w", None)  # no need for these weights now
        module.state.pop("wq", None) # no need for these weights now

    @classmethod
    def name(cls) -> str:
        return "de-quantize"
