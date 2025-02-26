import os
from dataclasses import dataclass, field
from typing import Dict, List, Union
from urllib.parse import urlparse

import safetensors
import torch

from ..utils.logger import setup_logger

logger = setup_logger()
LORA_MERGED_WEIGHT_PATHS = [None, ""]

# TODO FIX ME: cache of adapter tensors loaded from disk
adapter_load_cache = None

class Adapter():
    def __init__(self, rank: int, path: str = None):
        self.rank = rank
        self.path = path.lower().strip() if isinstance(path, str) else path

    def validate_path(self, local_only=False):
        if not self.path or not isinstance(self.path, str):
            raise ValueError("Adapter: `path` str is required.")

        if local_only:
            if self.path.startswith("http"):
                raise ValueError(f"Adapter: `path` str in this context must be a local os path: actual = `{self.path}`.")

    # override me
    def apply(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        pass

    # override me
    def post_init(self, weight_key: str, device: torch.device, **kwargs):
        pass

    # override me
    def optimize(self):
        pass

    # override me
    @classmethod
    def name(cls) -> List[str]:
        pass

    # override me
    @classmethod
    def parameter_keys(cls) -> [str]: # name of tensors/parameters in attribute key name
        pass


@dataclass
class Lora(Adapter):
    def __init__(self, rank: int, path: str = None, lora_A: torch.Tensor = None, lora_B: torch.Tensor = None):
        super().__init__(rank, path)

        self.lora_A = lora_A
        self.lora_B = lora_B

    @classmethod
    def name(cls) -> str:
        return "lora"

    @classmethod
    def parameter_keys(cls) -> List[str]:
        return ["lora_A", "lora_B"]

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        pass
        #logger.info("Adapter: optimize (compile)")
        #self.apply = torch_compile(self.apply, backend=backend, mode=mode, fullgraph=fullgraph)

    def apply(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        # original code
        # out = out + ((x @ self.lora_A) @ self.lora_B)

        # fix batch for lora
        # Some kernels do not reshape x, such as marlin / exllama / exllamav2.
        # out.dim() > x.dim() is used to exclude these kernels without additional processing
        if out.dim() > x.dim() and out.shape[0] > 1:
            out_orgi_shape = out.shape
            out = out.view(-1, out.shape[-1])
            out.add_((x @ self.lora_A) @ self.lora_B)
            out = out.view(out_orgi_shape)
            return out
        else:
            return out.add_((x @ self.lora_A) @ self.lora_B)

    def post_init(self, weight_key: str, device:torch.device, lora_A: torch.Tensor=None, lora_B: torch.Tensor=None):
        # self.register_buffer("lora_A", lora_A)
        # self.register_buffer("lora_B", lora_B)

        # we need since lora A/B weights may be merged into model tensors and not separate
        if lora_A is not None and lora_B is not None:
            # print(f"Adapter has preloaded lora_A and lora_B")
            self.lora_A, self.lora_B = lora_A, lora_B
            return

        global adapter_load_cache
        if adapter_load_cache is None:
            if os.path.isfile(self.path):
                lora_path = self.path
                logger.info(f"Adapter: Loading `{self.path}` tensors from disk")  # {adapter_load_cache}
            elif self.path.startswith("http"):
                from huggingface_hub import hf_hub_download
                result = self.parse_url(self.path)
                if len(result) == 3:
                    logger.info(f"Adapter: Downloading adapter weights from hf repo: `{result[0]}` revision: `{result[1]}` file: `{result[2]}`")
                    lora_path = hf_hub_download(repo_id=result[0], revision=result[1], filename=result[2])
                elif len(result) == 1:
                    logger.info(f"Adapter: Downloading adapter weights from uri = `{self.path}`")
                    import requests
                    response = requests.get(self.path, stream=True)
                    lora_path = "lora.safetensors"
                    with open(lora_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    raise Exception(f"Adapter: Lora path is invalid: `{self.path}`")
            else:
                from huggingface_hub import HfApi, hf_hub_download
                files = [f for f in HfApi().list_repo_files(self.path) if f in ["lora.safetensors", "eora_test.safetensors"]]

                if files:
                    lora_path = hf_hub_download(repo_id=self.path, filename=files[0])
                    # print(f"Adapter tensors loaded from `{self.path}`")
                else:
                    raise Exception(f"Adapter: There's no lora.safetensors or eora_test.safetensors on repo `{self.path}`")

            adapter_load_cache = safetensors.torch.load_file(lora_path)

        weight_key = weight_key.lower()

        # hack for HF Auto compat
        if not f"{weight_key}.lora_A.weight" in adapter_load_cache:
            weight_key = weight_key.removeprefix("model.")

        #print(f"loaded lora weight keys: {adapter_load_cache.keys()}")
        lora_A = adapter_load_cache.pop(f"{weight_key}.lora_A.weight").T
        lora_B = adapter_load_cache.pop(f"{weight_key}.lora_B.weight").T

        # since loder cache is singleton, we need to reset to None to ci loop tests can pass
        if len(adapter_load_cache) == 0:
            adapter_load_cache = None

        # print(f"Adapter: {self.name()}, loaded lora_A shape: {lora_A.shape}")
        # print(f"Adapter: {self.name()}, loaded lora_B shape: {lora_B.shape}")
        if lora_A.dtype != torch.float16 or lora_A.dtype != torch.float16:
            logger.warn(f"Adapter: `lora_A` and `lora_B` tensors should be of dtype = `torch.float16`: actual = `[{lora_A.dtype}, {lora_A.dtype}]`.")

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
        raise ValueError("Adapter: Invalid adapter config: `adapter`.")

    adapter_type = adapter.pop("name", None)
    if adapter_type is None:
        raise ValueError(f"Adapter: Invalid adapter class `{adapter_type}`: expected = `{ADAPTER_MAPPING}`.")

    adapterCls = ADAPTER_MAPPING.get(adapter_type)
    if adapterCls is None:
        raise ValueError(f"Adapter: Compatible adapters include `{ADAPTER_MAPPING.keys()}`: actual `{(adapter_type)}`.")

    try:
        adapterInstance = adapterCls(**adapter)
    except Exception:
        raise ValueError(f"Adapter: Invalid adapter config: `{adapter}`.")

    return adapterInstance
