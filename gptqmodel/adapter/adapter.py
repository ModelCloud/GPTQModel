# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple, Union

import pcre
import safetensors
import torch

from ..utils.logger import setup_logger
from .peft import LoraConfig
from .remote import resolve_checkpoint_adapter_path, resolve_path, validate_remote_pin


log = setup_logger()
LORA_MERGED_WEIGHT_PATHS = [None, ""]
HF_ADAPTER_FILE_NAME = "adapter_model.safetensors"
HF_ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
HF_ADAPTER_WEIGHT_KEY_PREFIX = "base_model.model."


def _rank_pattern_matches(pattern: str, weight_key: str, *, ignore_case: bool = False) -> bool:
    """Return whether one `rank_pattern` key selects `weight_key`.

    The checks mirror PEFT's documented order: an exact module name, then a
    module-name suffix, then a regex search. The pattern is matched exactly as
    written. Canonical module names are case-sensitive (MPT's `Wqkv`,
    Baichuan's `W_pack`) and regex escapes such as `\\S`, `\\D`, and `\\W` are
    meaningful, so the pattern text is never rewritten.

    `ignore_case` is the explicit legacy-export fallback. It folds only the
    literal comparisons and the regex engine's case flag; the pattern itself is
    passed through unchanged, so escapes keep their meaning.
    """

    if ignore_case:
        folded_key = weight_key.lower()
        folded_pattern = pattern.lower()
        if folded_key == folded_pattern or folded_key.endswith(folded_pattern):
            return True
        return pcre.compile(pattern, pcre.IGNORECASE).search(weight_key) is not None

    if weight_key == pattern or weight_key.endswith(pattern):
        return True
    return pcre.compile(pattern).search(weight_key) is not None


class AdapterCache():
    """Caches loaded adapter configs and tensors by source path."""

    cache: Dict[Hashable, Dict[str, Union[LoraConfig, Dict[str, torch.Tensor]]]] = {}

    @classmethod
    def get(cls, path: Hashable) -> Optional[Tuple[LoraConfig, Dict[str, torch.Tensor]]]:
        """Returns cached adapter config and weights for a path, if present."""

        data = cls.cache.get(path)
        if not data:
            return None
        else:
            return data["config"], data["weights"]

    @classmethod
    def reset(cls):
        """Clears the global adapter cache."""

        log.info("Adapter Cache: Resetting cache")
        cls.cache = {}

    @classmethod
    def add(cls, path: Hashable, config: LoraConfig, weights: Dict[str, torch.Tensor]):
        """Stores adapter config and weight tensors under the source path."""

        cls.cache[path] = {"config": config, "weights": weights}

    @classmethod
    def remove(cls, key):
        """Drops cached adapter state for a cache key if it exists."""

        cls.cache.pop(key, None)


class Adapter():
    """Base interface for runtime adapters applied on top of quantized layers."""

    def __init__(self, rank: int = None, path: str = None):
        """Initializes adapter identity and optional source location."""

        self.rank = rank # rank may be zero, when loading, and rank will be re-populated by loading saved LoraConfig file
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        self.path = path.strip() if isinstance(path, str) else path
        self._gptqmodel_remote_revision: Optional[str] = None
        self._gptqmodel_remote_sha256: Optional[str] = None
        self._gptqmodel_checkpoint_source: Optional[Tuple[str, str]] = None

    def validate_path(self, local=False):
        """Validates that the configured adapter path matches the expected source type."""

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
        """Applies the adapter contribution to a layer output tensor."""

        pass

    # override me
    def post_init(self, weight_key: str, device: torch.device, **kwargs):
        """Loads or finalizes adapter tensors for a specific weight key."""

        pass

    # override me
    def optimize(self):
        """Applies optional backend-specific optimizations after loading."""

        pass

    # override me
    @classmethod
    def name(cls) -> List[str]:
        """Returns the serialized adapter type name."""

        pass

    # override me
    @classmethod
    def parameter_keys(cls) -> [str]: # name of tensors/parameters in attribute key name
        """Lists tensor attribute names expected on the adapter instance."""

        pass


@dataclass
class Lora(Adapter):
    """LoRA adapter implementation backed by A/B projection matrices."""

    def __init__(self, rank: int, path: str = None, lora_A: torch.Tensor = None, lora_B: torch.Tensor = None):
        """Initializes the adapter with optional preloaded LoRA matrices."""

        super().__init__(rank, path)

        self.lora_A = lora_A
        self.lora_B = lora_B

    @classmethod
    def name(cls) -> str:
        """Returns the canonical adapter type used in serialized configs."""

        return "lora"

    @classmethod
    def parameter_keys(cls) -> List[str]:
        """Lists the tensor attributes that store the LoRA projection weights."""

        return ["lora_A", "lora_B"]

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        """Reserved hook for compiling the adapter path when enabled."""

        pass
        #logger.info("Adapter: optimize (compile)")
        #self.apply = torch_compile(self.apply, backend=backend, mode=mode, fullgraph=fullgraph)

    def apply(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Return the base output plus LoRA without modifying the base tensor."""

        # native quantized model/eora is float16 for gptq but for training, we may load the model as bfloat16 for accuracy
        if x.dtype != self.lora_A.dtype or x.device != self.lora_A.device:
            log.info.once(
                f"Adapter: Lora A/B auto changed from `{self.lora_A.dtype}` on `{self.lora_A.device}` "
                f"to `{x.dtype}` on `{x.device}` to match forward input."
            )
            self.lora_A = self.lora_A.to(device=x.device, dtype=x.dtype)
            self.lora_B = self.lora_B.to(device=x.device, dtype=x.dtype)

        update = (x @ self.lora_A) @ self.lora_B

        # fix batch for lora
        # Some kernels do not reshape x, such as marlin / exllama / exllamav2.
        # out.dim() > x.dim() is used to exclude these kernels without additional processing
        original_shape = None
        if out.dim() > x.dim() and out.shape[0] > 1:
            original_shape = out.shape
            out = out.view(-1, out.shape[-1])
        if torch.broadcast_shapes(out.shape, update.shape) != out.shape:
            raise RuntimeError("LoRA update would expand the base output shape")
        result = out + update
        return result.view(original_shape) if original_shape is not None else result

    def post_init(self, weight_key: str, device:torch.device, lora_A: torch.Tensor=None, lora_B: torch.Tensor=None):
        """Loads, caches, and materializes LoRA tensors for the target module."""

        # self.register_buffer("lora_A", lora_A)
        # self.register_buffer("lora_B", lora_B)

        # we need since lora A/B weights may be merged into model tensors and not separate
        if lora_A is not None and lora_B is not None:
            # print(f"Adapter has preloaded lora_A and lora_B")
            self.lora_A, self.lora_B = lora_A, lora_B
            return

        if self._gptqmodel_checkpoint_source is not None:
            checkpoint_path, manifest_path = self._gptqmodel_checkpoint_source
            self.path = resolve_checkpoint_adapter_path(manifest_path, checkpoint_path)

        remote_revision = getattr(self, "_gptqmodel_remote_revision", None)
        remote_sha256 = getattr(self, "_gptqmodel_remote_sha256", None)
        validate_remote_pin(remote_revision, remote_sha256)
        if remote_revision is None and isinstance(self.path, str) and os.path.isdir(self.path):
            self.path = os.path.abspath(self.path)
        cache_key = (self.path, remote_revision, remote_sha256)
        lora_cache = AdapterCache.get(cache_key)
        if lora_cache is None:
            # get lora config
            lora_cfg = LoraConfig.from_pretrained(
                path=self.path,
                filename=HF_ADAPTER_CONFIG_FILE_NAME,
                allow_remote=remote_revision is not None,
                revision=remote_revision,
            )
            lora_cfg.gptqmodel_path = self.path  # hack: save this

            if not isinstance(lora_cfg, LoraConfig):
                raise ValueError(f"Adapter: Expected `LoraConfig` in `{self.path}`, actual = `{lora_cfg}`")

            lora_path = resolve_path(
                self.path,
                HF_ADAPTER_FILE_NAME,
                allow_remote=remote_revision is not None,
                revision=remote_revision,
                expected_sha256=remote_sha256,
            )

            # save to adapter cache
            AdapterCache.add(cache_key, lora_cfg, safetensors.torch.load_file(lora_path))

            lora_cache = AdapterCache.get(cache_key)
            assert lora_cache is not None

        # lora_cache result is a tuple
        lora_cfg, lora_weights = lora_cache
        if self.rank is None:
            self.rank = lora_cfg.r
        elif self.rank != lora_cfg.r:
            raise ValueError(f"Adapter: `rank` must match `LoraConfig.r`, expected `{self.rank}`, actual = `{lora_cfg.r}`")

        def _extract_lora(module_key: str):
            suffix_a = f"{module_key}.lora_A.weight"
            suffix_b = f"{module_key}.lora_B.weight"
            found_a = found_b = None
            found_keys = []
            for k, v in lora_weights.items():
                if k.endswith(suffix_a):
                    found_a = v.T
                    found_keys.append(k)
                elif k.endswith(suffix_b):
                    found_b = torch.clone(v.T, memory_format=torch.contiguous_format)
                    found_keys.append(k)
            return found_a, found_b, found_keys

        # HF/PEFT safetensors keys are case-sensitive: MPT's `Wqkv` and
        # Baichuan's `W_pack` must match the module name exactly.
        lora_A, lora_B, pop_keys = _extract_lora(weight_key)
        if lora_A is None or lora_B is None:
            # Legacy GPT-QModel exports lowercased module names. Retry with the
            # historical spelling, but keep the canonical case for new exports.
            legacy_key = weight_key.lower()
            if legacy_key != weight_key:
                legacy_a, legacy_b, legacy_keys = _extract_lora(legacy_key)
                if legacy_a is not None and legacy_b is not None:
                    lora_A, lora_B, pop_keys = legacy_a, legacy_b, legacy_keys
                    weight_key = legacy_key

        if pop_keys:
            for k in pop_keys:
                lora_weights.pop(k) # releasee lora weights from cache memory

            # we have consumed all modules
            if len(lora_weights) == 0:
                AdapterCache.remove(cache_key)
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
        """Overrides the adapter rank when the config defines a matching rank pattern."""

        assert lora_cfg.rank_pattern is not None and weight_key is not None
        if lora_cfg.rank_pattern:
            for k, v in lora_cfg.rank_pattern.items():
                assert isinstance(k, str) and isinstance(v, int)
                assert v > 0 # check for invalid rank range
                # Match the pattern exactly as written: full name, then suffix,
                # then regex. Lowercasing here silently missed canonical
                # mixed-case module names and inverted regex escapes.
                matched = _rank_pattern_matches(k, weight_key)
                if not matched and (k != k.lower() or weight_key != weight_key.lower()):
                    # Legacy GPT-QModel exports lowercased module names. Retry
                    # case-insensitively without rewriting the pattern.
                    matched = _rank_pattern_matches(k, weight_key, ignore_case=True)
                if matched:
                    self.rank = v
                    log.info(f"Adapter: Base Lora `rank` = `{self.rank}` has been overridden by `{k}` due to dynamic `LoraConfig.rank_pattern` control.")
                    return True

        return False



    def to_dict(self):
        """Serializes the minimal adapter descriptor used by GPT-QModel."""

        return {
            "name": self.name(),
            "path": self._gptqmodel_checkpoint_source[1] if self._gptqmodel_checkpoint_source else self.path,
            "rank": self.rank
        }

ADAPTER_MAPPING = {Lora.name(): Lora}

# accept both Adapter cls instance or Dict()
def normalize_adapter(adapter:  Union[Dict, Adapter]):
    """Normalizes serialized adapter metadata into a concrete adapter instance."""

    if adapter is None:
        return None

    if isinstance(adapter, Adapter):
        return adapter

    if not isinstance(adapter, Dict):
        raise ValueError("Adapter: Invalid adapter config: `adapter`.")

    # Callers may reuse serialized adapter payloads across retries.
    adapter = dict(adapter)
    adapter_type = adapter.pop("name", None)
    if adapter_type is None:
        raise ValueError(f"Adapter: Invalid adapter class `{adapter_type}`: expected = `{ADAPTER_MAPPING}`.")

    adapterCls = ADAPTER_MAPPING.get(adapter_type)
    if adapterCls is None:
        raise ValueError(f"Adapter: Compatible adapters include `{ADAPTER_MAPPING.keys()}`: actual `{(adapter_type)}`.")

    # Serialized manifests are data-only: never allow non-tensor adapter weights.
    for tensor_key in adapterCls.parameter_keys() or []:
        if adapter.get(tensor_key) is not None and not isinstance(adapter[tensor_key], torch.Tensor):
            raise ValueError(f"Adapter: Invalid adapter config: `{adapter}`.")

    try:
        adapterInstance = adapterCls(**adapter)
    except Exception:
        raise ValueError(f"Adapter: Invalid adapter config: `{adapter}`.")

    return adapterInstance
