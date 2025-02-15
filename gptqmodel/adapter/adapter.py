import os
from dataclasses import dataclass, field
from typing import Dict, Union
from urllib.parse import urlparse

import safetensors
import torch

LORA_MERGED_WEIGHT_PATHS = [None, ""]

# TODO FIX ME: cache of adapter tensors loaded from disk
adapter_load_cache = None

@dataclass
class Adapter():
    path: str
    rank: int

    # override me
    def apply(self, x: torch.Tensor, out: torch.Tensor):
        pass

    # override me
    def post_init(self, weight_key: str, device: torch.device, **kwargs):
        pass

    # override me
    @classmethod
    def name(cls) -> str:
        pass


@dataclass
class Lora(Adapter):
    path: str = field(default=None)
    rank: int = field(default=256, metadata={"choices": [32, 64, 128, 256, 512]})

    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None

    @classmethod
    def name(cls) -> str:
        return "lora"

    def apply(self, x: torch.Tensor, out: torch.Tensor):
        #out = out + ((x @ self.lora_A) @ self.lora_B)
        return out.add_((x @ self.lora_A) @ self.lora_B)

    def post_init(self, weight_key: str, device:torch.device, lora_A: torch.Tensor=None, lora_B: torch.Tensor=None):
        # we need since lora A/B weights may be merged into model tensors and not separate
        if lora_A is not None and lora_B is not None:
            print(f"Adapter has preloaded lora_A and lora_B")
            self.lora_A, self.lora_B = lora_A, lora_B
            return

        global adapter_load_cache
        if adapter_load_cache is None:
            if os.path.isfile(self.path):
                lora_path = self.path
                print(f"loading adapter `{self.path}` tensors from disk")  # {adapter_load_cache}
            elif self.path.startswith("http"):
                from huggingface_hub import hf_hub_download
                result = self.parse_url(self.path)
                if len(result) == 3:
                    print(f"downloading adapter from huggingface. repo: {result[0]} revision: {result[1]} file: {result[2]}")
                    lora_path = hf_hub_download(repo_id=result[0], revision=result[1], filename=result[2])
                elif len(result) == 1:
                    print(f"downloading adapter from link `{self.path}`")
                    import requests
                    response = requests.get(self.path, stream=True)
                    lora_path = "lora.safetensors"
                    with open(lora_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    raise Exception(f"lora path is invalid: `{self.path}`")
            else:
                from huggingface_hub import HfApi, hf_hub_download
                files = [f for f in HfApi().list_repo_files(self.path) if f in ["lora.safetensors", "eora_test.safetensors"]]

                if files:
                    lora_path = hf_hub_download(repo_id=self.path, filename=files[0])
                    print(f"Adapter tensors loaded from `{self.path}`")
                else:
                    raise Exception(f"There's no lora.safetensors or eora_test.safetensors on repo `{self.path}`")

            adapter_load_cache = safetensors.torch.load_file(lora_path)

        lora_A = adapter_load_cache.pop(f"{weight_key}.lora_A.weight").T
        lora_B = adapter_load_cache.pop(f"{weight_key}.lora_B.weight").T

        # since loder cache is singleton, we need to reset to None to ci loop tests can pass
        if len(adapter_load_cache) == 0:
            adapter_load_cache = None

        print(f"Adapter: {self.name()}, loaded lora_A shape: {lora_A.shape}")
        print(f"Adapter: {self.name()}, loaded lora_B shape: {lora_B.shape}")
        if lora_A.dtype != torch.float16 or lora_A.dtype != torch.float16:
            print(
                f"Warning: lora_A and lora_B tensors should be `torch.float16`: actual = `[{lora_A.dtype}, {lora_A.dtype}]`.")

        self.lora_A = lora_A.to(device=device, dtype=torch.float16)
        self.lora_B = lora_B.to(device=device, dtype=torch.float16)

        #print(f"Adapter: lora_A {lora_A.shape}: `{lora_B}`")
        #print(f"Adapter: lora_B {lora_B.shape}: `{lora_B}`")

    def parse_url(self, url: str):
        parsed_url = urlparse(url)

        if parsed_url.netloc.endswith("huggingface.co") or parsed_url.netloc.endswith("hf.co"):
            parts = parsed_url.path.strip("/").split("/")

            if "blob" in parts:
                idx = parts.index("blob")
                repo_id = "/".join(parts[:idx])
                rev = parts[idx + 1]
                filename = parts[idx + 2].split("?")[0] # remove ?download=true
                return [repo_id, rev, filename]
        else:
            return [url]
        return []

    def to_dict(self):
        return {
            "name": self.name(),
            "path": self.path,
            "rank": self.rank
        }

ADAPTER_MAPPING = {Lora.name(): Lora}

# accept both Adapter cls instance or Dict()
def normalize_adapter(adapter:  Union[Dict, Adapter]):
    if adapter is None:
        return None

    if isinstance(adapter, Adapter):
        return adapter

    if not isinstance(adapter, Dict):
        raise ValueError("Invalid adapter config: `adapter`.")

    adapter_type = adapter.pop("name", None)
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
