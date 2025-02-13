from typing import Dict, List

from torch import Tensor
from torch.nn import Module


class LoopProcessor:
    # called first
    def preprocess(self, module: Module):
        pass

    # called after every module generate
    # may be called multiple times due to batch
    def receive_inputs(self, inputs: List[Tensor]):
        pass

    # do work and return processor state which will be merged into looper state
    def process(self, module: Module, state: Dict[str, ]):
        pass

    # step after `process` and before post_process generate()
    def post_process(self, module: Module, state: Dict[str,]):
        pass

    # last step, after all loop processor is called
    def finalize(self, module:Module, state: Dict[str,]):
        pass
