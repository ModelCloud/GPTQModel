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

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..utils.logger import setup_logger
from ..utils.torch import torch_compile

log = setup_logger()

class DequantizeProcessor(LoopProcessor):
    def __init__(self, quantized_modules: Dict[str, TorchQuantLinear]):
        super().__init__(tokenizer=None, qcfg=None, calibration_dataset=None, calibration_dataset_concat_size=None,
                         prepare_dataset_func=None, batch_size=1,
                         logger_board="", require_fwd=True)

        self.quantized_modules = quantized_modules

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = None
        self.num_batches = 0

    # de-quantize weights
    def process(self, module: NamedModule):
        device = module.weight.device
        w = module.weight.data

        # TODO fix num_itr param..need to calculate this before dequant
        m = self.quantized_modules.pop(module.full_name)
        m.dequantize_weight = torch_compile(m.dequantize_weight)
        wq = m.dequantize_weight().T.to(device=device)

        module.state.update({
            "w": w,
            "wq": wq,
        })

    def submodule_finalize(self, module: NamedModule):
        module.state.pop("w", None)  # no need for these weights now
        module.state.pop("wq", None) # no need for these weights now

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        return False

    @classmethod
    def name(cls) -> str:
        return "de-quantize"
