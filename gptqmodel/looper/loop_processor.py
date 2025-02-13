from typing import Dict, List

from torch import Tensor
from torch.nn import Module

# LoopProcessor is a singleton(), not per module instance
class LoopProcessor:
    inputs_cache: List[Tensor] = []

    def __init__(self, calibration_data):
        self.calibration_data = calibration_data

    # called first
    def preprocess(self, module: Module):
        pass

    # called after every module generate
    # may be called multiple times due to batch
    def receive_inputs(self, inputs: Tensor):
        self.inputs_cache += inputs

    # do work and return processor state which will be merged into looper state
    def process(self, module: Module, state: Dict[str, ]):
        pass

    # step after `process` and before post_process generate()
    def post_process(self, module: Module, state: Dict[str,]):
        pass

    def clear_input(self):
        self.inputs_cache = []

    # last step, after all loop processor is called
    def finalize(self, module:Module, state: Dict[str,]):
        pass
