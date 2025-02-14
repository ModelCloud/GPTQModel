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

from typing import Callable, List, Tuple

import torch
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models import BaseGPTQModel
from torch import Tensor
from torch.nn import Module


# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    def __init__(self, calibration_dataset, qcfg: QuantizeConfig, logger_board:str="", require_fwd: bool = True):
        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = {}
        self.calibration_dataset = calibration_dataset
        self.qcfg = qcfg

        # if processor require fwd generate and hooks, set this to true
        # looper should bypass generate + hooks if this is false
        self.require_fwd = require_fwd

        self.logger_task = None

    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        pass

    def receive_input_cache(self, input_cache: InputCache):
        self.inputs_cache = input_cache

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_input(self, layer_input: List[Tensor]):
        self.inputs_cache.layer_inputs.append(layer_input)

    def clear_cache_data(self):
        self.tasks = {}
        del self.inputs_cache.layer_inputs
        self.inputs_cache.layer_inputs = []

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        pass

    # do work and return processor.self state which will updated/merged
    def process(self, module: NamedModule):
        pass

    # step after `process` and before post_process generate()
    def post_process(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    def submodule_finalize(self, module: NamedModule):
        pass

    # last step, after all loop processor is called
    def model_finalize(self, gptq_model: BaseGPTQModel, **kwargs):
        pass
