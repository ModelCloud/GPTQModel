
import torch


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module
        self.name = name
        self.full_name = full_name
        self.layer_index = layer_index

        self.state = {}

    def __getattr__(self, item: str):
        if item == "name":
            return self.name
        elif item == "full_name":
            return self.full_name
        elif item == "layer_index":
            return self.layer_index

        return getattr(self.module, item)
