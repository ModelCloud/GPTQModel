
import torch
import transformers
from torch import nn


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module # wrapped module
        self.name = name # module name
        self.full_name = full_name # module full name (path) within model
        self.layer_index = layer_index # layerid in a repeating layer, if in outside layer, this info may be fake
        self.state = {} # state is dict to store all temp data used in processor

        # store original in/out features since weight.data will changed later on
        if isinstance(module.module, nn.Linear):
            in_features = module.module.in_features
            out_features = module.module.out_features
        elif isinstance(module.module, nn.Conv2d):
            in_features = module.module.in_channels
            out_features = module.module.out_channels
        elif isinstance(module.module, transformers.pytorch_utils.Conv1D):
            in_features = module.module.weight.shape[0]
            out_features = module.module.weight.shape[1]
        else:
            raise NotImplementedError(f"Unsupported module.module type: `{type(module.module)}`")

        self.state.update({
            "in_features": in_features,
            "out_features": out_features,
        })

    def __getattr__(self, name: str):
        if name in ["module", "name", "full_name", "layer_index", "state"]:
            return getattr(self, name)

        return getattr(self.module, name)
