
import torch


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module
        self.name = name
        self.full_name = full_name
        self.layer_index = layer_index

    def __getattr__(self, item):
        try:
            if item == "name":
                return self.name
            elif item == "full_name":
                return self.full_name
            elif item == "layer_index":
                return self.layer_index

            return self.module.__getattr__(item)
        except Exception:
            return getattr(self.model, item)