import threading
from typing import List, Optional

import torch
# move base_module tensors to disk
from accelerate import disk_offload
from torch import nn

from .device import CPU
from .structure import print_module_tree

_lock = threading.Lock()

def get_module_fullname(model: torch.nn.Module, module: torch.nn.Module) -> str:
    for name, mod in model.named_modules():
        if mod is module:
            return name  # dotted path like "model.embed_tokens" or "model.layers.0.self_attn.q_proj"

    raise Exception("module not found in model")

def set_submodule(root: torch.nn.Module, path: str, new_mod: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    m = root
    for part in path.split("."):
        m = getattr(m, part)
    return m

def is_meta_module(m: nn.Module) -> bool:
    for p in m.parameters(recurse=True):
        if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device.type == "meta"):
            return True
    for b in m.buffers(recurse=True):
        if hasattr(b, "device") and b.device.type == "meta":
            return True
    return False

def offload_to_disk(module: List[str] | nn.Module, model: Optional[nn.Module] ):
    assert module is not None
    assert model is not None

    with _lock:
        if isinstance(module, List):
            for name in module:
                m = get_submodule(model, name)
                full_name = get_module_fullname(model=model, module=m)
                _offload_disk(module=m, name=full_name)
        else:
            full_name = get_module_fullname(model=model, module=module)
            _offload_disk(module=module, name=full_name)

def _offload_disk(module: nn.Module, name: str):
    if is_meta_module(module):
        print(f"[skip] '{name}' is on meta; leaving as-is")
        return

    # print(f"device_map base_modules: {device_map}")

    _ = disk_offload(
        module,
        # device_map={ "" : "disk" },  # only touch this subtree
        offload_dir=f"offload/{name}",
        offload_buffers=True,  # needed for buffers
        execution_device=CPU,
    )

    print("offload_disk: list item tree")
    print_module_tree(module)
