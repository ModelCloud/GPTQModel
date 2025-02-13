
import torch


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__(module)

        self.name = name
        self.full_name = full_name
        self.layer_index = layer_index