
import torch


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module
        self.name = name
        self.full_name = full_name
        self.layer_index = layer_index
        self.state = {} # state is dict to store all temp data used in processor

    def __getattr__(self, name: str):
        if name in ["name", "full_name", "layer_index", "state"]:
            return getattr(self, name)

        return getattr(self.module, name)
