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

import copy
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.gptq_processor import GPTQProcessor
from ..looper.named_module import NamedModule
from ..quantization import GPTQv2
from ..quantization.config import QUANT_METHOD, QuantizeConfig
from ..utils.logger import setup_logger

log = setup_logger()


class GPTQv2Processor(GPTQProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True, retain_w: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration_dataset=calibration_dataset,
                         calibration_dataset_concat_size=calibration_dataset_concat_size,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         logger_board=logger_board, require_fwd=require_fwd)


    def preprocess(self, module: NamedModule, buffered_fwd: bool):
        # entire module is skipped
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:
            return

        qcfg_clone = copy.deepcopy(self.qcfg)

        # dynamic overrides
        if self.qcfg.dynamic is not None:
            qcfg_clone.bits = self.qcfg.dynamic_get(module.full_name, "bits", qcfg_clone.bits)
            qcfg_clone.sym = self.qcfg.dynamic_get(module.full_name, "sym", qcfg_clone.sym)
            qcfg_clone.mse = self.qcfg.dynamic_get(module.full_name, "mse", qcfg_clone.mse)

            qcfg_clone.group_size = self.qcfg.dynamic_get(module.full_name, "group_size", qcfg_clone.group_size)
            qcfg_clone.desc_act = self.qcfg.dynamic_get(module.full_name, "desc_act", qcfg_clone.desc_act)
            qcfg_clone.damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", qcfg_clone.damp_percent)
            qcfg_clone.static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", qcfg_clone.static_groups)

        tmp = GPTQv2(module=module, qcfg=qcfg_clone)

        # models like DeepSeek v3/r1 has > 256 $ of sub-modules per layer
        # use buffered mode go vram don't explode: gptq needs to store fwd inputs per each layer fwd
        # all sub-modules within a single layer needs to store all the inputs.
        # deepseek has massive # of sub-modules per layer, causing vram pressure
        # buffered mode is slower due to gpu<->cpu movement
        if buffered_fwd:  # TODO tweak this number for masive MoE
            log.info(f"Experimental: enabling fwd buffered mode for: `{module.name}`")
            tmp.fwd_inputs_buffered = True

        tmp.quantizer.configure(
            perchannel=True,
        )
        self.tasks[module.name] = tmp

    @classmethod
    def name(cls) -> str:
        return "gptqv2"
