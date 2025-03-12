
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import safetensors
import torch

from ..utils.logger import setup_logger
from .peft import LoraConfig
from .remote import resolve_path

log = setup_logger()
LORA_MERGED_WEIGHT_PATHS = [None, ""]
HF_ADAPTER_FILE_NAME = "adapter_model.safetensors"
HF_ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
HF_ADAPTER_WEIGHT_KEY_PREFIX = "base_model.model."


class AdapterCache():
    cache: Dict[str, Dict[str, Union[LoraConfig, torch.Tensor]]] = {}  # first level key is `path`, second level keys [ `config` = LoraConfig, `weights` = Dict[str, Tensors]

    @classmethod
    def get(cls, path: str) -> Optional[Tuple[LoraConfig, Dict[str, torch.Tensor]]]:
        data = cls.cache.get(path)
        if not data:
            return None
        else:
            return data["config"], data["weights"]

    @classmethod
    def reset(cls):
        log.info("Adapter Cache: Resetting cache")
        cls.cache = {}

    @classmethod
    def add(cls, path: str, config: LoraConfig, weights: Dict[str, torch.Tensor]):
        cls.cache[path] = {"config": config, "weights": weights}

    @classmethod
    def remove(cls, path):
        cls.cache.pop(path, None)


class Adapter():
    def __init__(self, rank: int = None, path: str = None):
        self.rank = rank # rank may be zero, when loading, and rank will be re-populated by loading saved LoraConfig file
        self.path = path.strip() if isinstance(path, str) else path

    def validate_path(self, local=False):
        if not self.path or not isinstance(self.path, str):
            raise ValueError("Adapter: `path` str is required.")

        # path should not be a file but a directory
        if self.path.endswith(".safetensors"):
            raise ValueError(
                f"Adapter: `path` must be a directory path or repo depending if you are saving (directory path) or loading (repo): actual = `{self.path}`")

        if local:
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

        # native quantized model/eora is float16 for gptq but for training, we may load the model as bfloat16 for accuracy
        if x.dtype != self.lora_A.dtype:
            log.info.once(f"Adapter: Lora A/B auto changed from `{self.lora_A.dtype}` to `{x.dtype}` to match forward input dtype.")
            self.lora_A = self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_B = self.lora_B.to(device=x.device, dtype=x.dtype)

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

        lora_cache = AdapterCache.get(self.path)
        if lora_cache is None:
            # get lora config
            lora_cfg = LoraConfig.from_pretrained(path=self.path, filename=HF_ADAPTER_CONFIG_FILE_NAME)
            lora_cfg.gptqmodel_path = self.path  # hack: save this

            if not isinstance(lora_cfg, LoraConfig):
                raise ValueError(f"Adapter: Expected `LoraConfig` in `{self.path}`, actual = `{lora_cfg}`")

            if self.rank is None:
                self.rank = lora_cfg.r
            else:
                if self.rank != lora_cfg.r:
                    raise ValueError(f"Adapter: `rank` must match `LoraConfig.r`, expected `{self.rank}`, actual = `{lora_cfg.r}`")

            lora_path = resolve_path(self.path, HF_ADAPTER_FILE_NAME)

            # save to adapter cache
            AdapterCache.add(self.path, lora_cfg, safetensors.torch.load_file(lora_path))

            lora_cache = AdapterCache.get(self.path)
            assert lora_cache is not None

        # lora_cache result is a tuple
        lora_cfg, lora_weights = lora_cache

        weight_key = weight_key.lower()

        # hack for HF Auto compat
        lora_A_weight_key = f"{weight_key}.lora_A.weight"
        lora_B_weight_key = f"{weight_key}.lora_B.weight"

        
        pop_keys = []
        for k, v in lora_weights.items():
            if k.endswith(lora_A_weight_key):
                lora_A = v.T
                pop_keys.append(k)
            elif k.endswith(lora_B_weight_key):
                lora_B = torch.clone(v.T, memory_format=torch.contiguous_format)
                pop_keys.append(k)

        if pop_keys:
            for k in pop_keys:
                lora_weights.pop(k) # releasee lora weights from cache memory

            # we have consumed all modules
            if len(lora_weights) == 0:
                AdapterCache.remove(self.path)
                log.info("Adapter: Consumed all Lora weights")

        else:
            log.warn(f"Adapter: Lora weights not found for `{weight_key}`")

        assert lora_A is not None and lora_B is not None, f"Adapter: `lora_A` and `lora_B` must both be present in the weights: actual = `{lora_A}` and `{lora_B}`"

        # check for rank override from base config
        self.dynamic_rank_override(lora_cfg=lora_cfg, weight_key=weight_key)

        # # since loder cache is singleton, we need to reset to None to ci loop tests can pass
        # if len(lora_weights) == 0:
        #     adapter_load_cache = None

        # print(f"Adapter: {self.name()}, loaded lora_A shape: {lora_A.shape}")
        # print(f"Adapter: {self.name()}, loaded lora_B shape: {lora_B.shape}")
        if lora_A.dtype not in [torch.float16, torch.bfloat16] or lora_B.dtype not in [torch.float16, torch.bfloat16]:
            log.warn.once(f"Adapter: `lora_A` and `lora_B` tensors should be of dtype = [torch.float16, torch.bfloat16]: actual = `[{lora_A.dtype}, {lora_B.dtype}]`.")

        # TODO: if a/b are float32, we will convert to model dtype in first forward pass
        # safe to downcast from float64
        self.lora_A = lora_A.to(device=device, dtype=torch.float32 if lora_A.dtype == torch.float64 else lora_A.dtype)
        self.lora_B = lora_B.to(device=device, dtype=torch.float32 if lora_B.dtype == torch.float64 else lora_B.dtype)

        #print(f"Adapter: lora_A {lora_A.shape}: `{lora_B}`")
        #print(f"Adapter: lora_B {lora_B.shape}: `{lora_B}`")

    def dynamic_rank_override(self, lora_cfg: LoraConfig, weight_key: str) -> bool:
        assert lora_cfg.rank_pattern is not None and weight_key is not None
        if lora_cfg.rank_pattern:
            for k, v in lora_cfg.rank_pattern.items():
                assert isinstance(k, str) and isinstance(v, int)
                k = k.lower()
                assert v > 0 # check for invalid rank range
                # first do string full match, then suffix match, then regex match
                if weight_key == k or k.endswith(weight_key) or re.match(k, weight_key):
                    self.rank = v
                    log.info(f"Adapter: Base Lora `rank` = `{self.rank}` has been overridden by `{k}` due to dynamic `LoraConfig.rank_pattern` control.")
                    return True

        return False



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
