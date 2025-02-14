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
    def __init__(self, calibration_dataset, qcfg: QuantizeConfig):
        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = []
        self.calibration_dataset = calibration_dataset
        self.qcfg = qcfg


    # called first
    def preprocess(self, module: NamedModule, **kwargs):
        pass

    def receive_input_cache(self, input_cache: InputCache):
        self.inputs_cache = input_cache

    # called after every module generate
    # may be called multiple times due to batch
    def receive_layer_input(self, layer_input: List[Tensor]):
        self.inputs_cache.layer_inputs += layer_input

    def clear_layer_inputs(self):
        del self.inputs_cache.layer_inputs
        self.inputs_cache.layer_inputs = []

    def create_task(self, name: str):
        pass

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
