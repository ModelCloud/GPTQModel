import os
from dataclasses import dataclass, field
from typing import Dict, Union

import safetensors
import torch

# TODO FIX ME: cache of adapter tensors loaded from disk
adapter_load_cache = None

@dataclass
class Adapter():
    name: str
    path: str
    rank: int

    # override me
    def apply(self, x: torch.Tensor, out: torch.Tensor):
        pass

    # override me
    def post_init(self, weight_key: str, device: torch.device):
        pass


@dataclass
class Lora(Adapter):
    name: str = "lora"
    path: str = field(default=None)
    rank: int = field(default=256, metadata={"choices": [32, 64, 128, 256, 512]})

    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None

    def apply(self, x: torch.Tensor, out: torch.Tensor):
        #out = out + ((x @ self.lora_A) @ self.lora_B)
        return out.add_((x @ self.lora_A) @ self.lora_B)

    def post_init(self, weight_key: str, device:torch.device):
        global adapter_load_cache
        if adapter_load_cache is None:
            if os.path.isfile(self.path):
                adapter_load_cache = safetensors.torch.load_file(self.path)
                print(f"Adapter `{self.path}` tensors loaded from disk")  # {adapter_load_cache}
            else:
                # TODO FIX ME add hf.co/huggingface.co download support
                raise Exception("Need to add HF support")

        lora_A = adapter_load_cache.pop(f"{weight_key}.lora_A.weight").T
        lora_B = adapter_load_cache.pop(f"{weight_key}.lora_B.weight").T

        # since loder cache is singleton, we need to reset to None to ci loop tests can pass
        if len(adapter_load_cache) == 0:
            adapter_load_cache = None

        print(f"Adapter: {self.name}, loaded lora_A shape: {lora_A.shape}")
        print(f"Adapter: {self.name}, loaded lora_B shape: {lora_B.shape}")
        if lora_A.dtype != torch.float16 or lora_A.dtype != torch.float16:
            print(
                f"Warning: lora_A and lora_B tensors should be `torch.float16`: actual = `[{lora_A.dtype}, {lora_A.dtype}]`.")

        self.lora_A = lora_A.to(device=device, dtype=torch.float16)
        self.lora_B = lora_B.to(device=device, dtype=torch.float16)

        #print(f"Adapter: lora_A {lora_A.shape}: `{lora_B}`")
        #print(f"Adapter: lora_B {lora_B.shape}: `{lora_B}`")

    def to_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "rank": self.rank
        }

ADAPTER_MAPPING = {"lora": Lora}

# accept both Adapter cls instance or Dict()
def normalize_adapter(adapter:  Union[Dict, Adapter]):
    if adapter is None:
        return None

    if isinstance(adapter, Adapter):
        return adapter

    if not isinstance(adapter, Dict):
        raise ValueError("Invalid adapter config: `adapter`.")

    adapter_type = adapter.get("name")
    if adapter_type is None:
        raise ValueError(f"Invalid adapter class `{adapter_type}`: expected = `{ADAPTER_MAPPING}`.")

    adapterCls = ADAPTER_MAPPING.get(adapter_type)
    if adapterCls is None:
        raise ValueError(f"QuantizeConfig.extension only accept `{ADAPTER_MAPPING.keys()}`: actual `{(adapter_type)}`.")

    try:
        adapterInstance = adapterCls(**adapter)
    except Exception:
        raise ValueError(f"Invalid adapter config: `{adapter}`.")

    return adapterInstance
