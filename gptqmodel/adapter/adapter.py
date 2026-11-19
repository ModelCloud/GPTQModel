# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import numbers
import os
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Tuple, Union

import pcre
import safetensors
import torch

from ..utils.logger import setup_logger
from .peft import LoraConfig
from .remote import resolve_checkpoint_adapter_path, resolve_path, validate_remote_pin
from .targeting import module_matches_targets


log = setup_logger()
LORA_MERGED_WEIGHT_PATHS = [None, ""]
HF_ADAPTER_FILE_NAME = "adapter_model.safetensors"
HF_ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"
HF_ADAPTER_WEIGHT_KEY_PREFIX = "base_model.model."


def _local_adapter_generation(path) -> Optional[Tuple]:
    """Return a cheap signature of a local adapter's files.

    An unpinned local adapter is cached by path, but the files can be rewritten
    in place (re-exporting to the same directory). Stat the two artifact files
    so an overwrite starts a new cache entry instead of serving the previous
    tensors (#5423).
    """

    try:
        path = os.fspath(path)
    except TypeError:
        return None
    if not isinstance(path, str) or not os.path.isdir(path):
        return None
    generation = []
    for filename in (HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME):
        try:
            stat = os.stat(os.path.join(path, filename))
        except OSError:
            generation.append((filename, None, None, None))
        else:
            generation.append((filename, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    return tuple(generation)


def _rank_pattern_matches(pattern: str, weight_key: str, *, ignore_case: bool = False) -> bool:
    """Return whether one `rank_pattern` key selects `weight_key`.

    This is PEFT's exact rule (``peft.utils.other.get_pattern_key``):
    ``re.match(rf"(.*\\.)?({pattern})$", weight_key)``. A bare suffix therefore
    has to start after a dot, so ``q_proj`` matches ``...self_attn.q_proj`` but
    ``1.self_attn.q_proj`` does not match layer 11. The pattern is matched
    exactly as written. Canonical module names are case-sensitive (MPT's
    `Wqkv`, Baichuan's `W_pack`) and regex escapes such as `\\S`, `\\D`, and
    `\\W` are meaningful, so the pattern text is never rewritten.

    `ignore_case` is the explicit legacy-export fallback. It folds only the
    regex engine's case flag; the pattern itself is passed through unchanged,
    so escapes keep their meaning.
    """

    flags = pcre.IGNORECASE if ignore_case else 0
    return pcre.compile(rf"(.*\.)?({pattern})$", flags).match(weight_key) is not None


def _pattern_override_candidates(
    patterns: Dict[str, object], weight_key: str
) -> List[Tuple[str, object]]:
    """List ``(pattern, value)`` in PEFT's precedence order for ``weight_key``.

    Exact, case-sensitive matches come first, in declaration order, because
    that is the order PEFT itself applies. The legacy case-insensitive fallback
    runs only when no pattern matched exactly: trying it per pattern let an
    earlier lowercase rule capture a later rule's canonical mixed-case module
    (`alpha_pattern={"proj": 8, "Proj": 2}` applied 8 to `block.Proj`, #5422).

    The fallback keeps its old reach: it applies only when the pattern or the
    module name is written in lowercase, which is what a legacy GPT-QModel
    export produces. A canonical pattern is never case-folded onto a canonical
    name.
    """

    exact: List[str] = []
    folded: List[str] = []
    for pattern in patterns:
        if _rank_pattern_matches(pattern, weight_key):
            exact.append(pattern)
        elif pattern == pattern.lower() or weight_key == weight_key.lower():
            if _rank_pattern_matches(pattern, weight_key, ignore_case=True):
                folded.append(pattern)
    return [(pattern, patterns[pattern]) for pattern in exact + folded]


# PEFT layouts whose tensors the LoRA forward cannot consume. A checkpoint that
# carries them but no `lora_A`/`lora_B` would attach nothing and run as the bare
# model, so they are rejected by name too (the config `peft_type` check is the
# primary guard; this catches a config that lies or predates the field).
_UNSUPPORTED_PEFT_TENSOR_MARKERS = (
    (".ia3_l", "IA3"),
    (".hada_", "LoHa"),
    (".lokr_", "LoKr"),
    (".oft_blocks", "OFT"),
    (".boft_blocks", "BOFT"),
    (".vera_lambda_", "VeRA"),
    (".lora_E", "AdaLoRA"),
)


def _reject_unsupported_adapter_tensors(weights: Dict[str, torch.Tensor]) -> None:
    """Fail closed on adapter tensors this runtime cannot apply.

    The LoRA path consumes only `lora_A` / `lora_B`. A checkpoint that also
    carries DoRA magnitude vectors, `trainable_token_indices` deltas, trained
    biases, or `modules_to_save` parameters would have those tensors dropped, so
    the adapter would load and run as a different (plain LoRA) model. Rejecting
    them keeps the load honest instead of silently wrong.
    """

    for key in weights:
        for marker, feature in _UNSUPPORTED_PEFT_TENSOR_MARKERS:
            if marker in key:
                raise ValueError(
                    f"Adapter: {feature} adapter tensors are not supported (`{key}`); "
                    "only LoRA `lora_A`/`lora_B` can be loaded"
                )
        if ".lora_embedding_" in key:
            # PEFT writes `lora_embedding_A`/`lora_embedding_B` for
            # `target_modules=["embed_tokens"]`. The runtime attaches only
            # `lora_A`/`lora_B` to linear layers, so the embedding delta used to
            # be dropped and the model ran as the bare checkpoint (#5508).
            raise ValueError(
                "Adapter: LoRA on an embedding layer is not supported "
                f"(`{key}`); PEFT stores those tensors as `lora_embedding_A`/"
                "`lora_embedding_B`, which this runtime cannot apply. Fold the "
                "embedding delta into the base weights, or drop `embed_tokens` "
                "from the adapter targets."
            )
        if ".lora_magnitude_vector" in key:
            raise ValueError(
                f"Adapter: DoRA is not supported (`lora_magnitude_vector` tensor found: `{key}`)"
            )
        if ".modules_to_save." in key:
            raise ValueError(f"Adapter: `modules_to_save` is not supported (`{key}`)")
        if ".trainable_tokens_delta" in key:
            raise ValueError(
                "Adapter: `trainable_token_indices` is not supported "
                f"(`trainable_tokens_delta` tensor found: `{key}`); "
                "load a plain LoRA adapter or fold the trained token deltas into the base model"
            )
        if key.endswith(".bias"):
            raise ValueError(f"Adapter: adapter bias tensors are not supported (`{key}`)")


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
        """Stores adapter config and weight tensors under the source path.

        The cached ``weights`` mapping is the immutable source of truth. A load
        records what it consumed in a separate ``consumed`` set instead of
        deleting entries, because one model load may legitimately consume only a
        subset of the adapter's target modules.
        """

        cls.cache[path] = {"config": config, "weights": weights, "consumed": set()}

    @classmethod
    def consume(cls, path: Hashable, keys: Iterable[str]) -> bool:
        """Record ``keys`` as consumed and report whether the entry is now spent.

        Consumption is recorded, never applied to the cached tensors: a second
        model load of the same adapter must still see every module. The entry is
        released once every cached weight has been consumed by some load.
        """

        data = cls.cache.get(path)
        if not data:
            return False
        consumed = data.setdefault("consumed", set())
        consumed.update(keys)
        return len(consumed) >= len(data.get("weights") or {})

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

    def targets_module(self, module_name: str) -> bool:
        """Whether this adapter applies to ``module_name``.

        The base adapter has no target metadata, so it keeps the historical
        attach-to-every-module behavior. Subclasses with a PEFT config override
        this to honour ``target_modules`` / ``exclude_modules``.
        """

        return True

    def has_weights(self, module_name: str) -> bool:
        """Whether the adapter source carries A and B tensors for ``module_name``.

        The base adapter has no tensor metadata; subclasses that can inspect
        their source override this. Target matching alone is not enough when a
        PEFT export only contains the layers it transformed.
        """

        return True

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
        self.lora_scale = 1.0
        # Whether ``lora_scale`` was multiplied into ``lora_B`` at load time.
        # ``apply()`` multiplies the product when it was not.
        self._lora_scale_folded = True
        self._lora_config: Optional[LoraConfig] = None

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
        if not self._lora_scale_folded:
            # Folding overflowed (or flushed a value) in the adapter dtype, so
            # the raw B stays stored and the effective scale is applied to the
            # product, which is the PEFT order (#5426).
            update = update * self.lora_scale

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

    def _resolve_source(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve the adapter source path and its explicit remote pin."""

        if self._gptqmodel_checkpoint_source is not None:
            checkpoint_path, manifest_path = self._gptqmodel_checkpoint_source
            self.path = resolve_checkpoint_adapter_path(manifest_path, checkpoint_path)

        remote_revision = getattr(self, "_gptqmodel_remote_revision", None)
        remote_sha256 = getattr(self, "_gptqmodel_remote_sha256", None)
        validate_remote_pin(remote_revision, remote_sha256)
        if remote_revision is None and isinstance(self.path, str) and os.path.isdir(self.path):
            self.path = os.path.abspath(self.path)
        return self.path, remote_revision, remote_sha256

    def _adapter_cache_key(self, path, remote_revision, remote_sha256):
        """Build the cache key, adding a local artifact generation when unpinned.

        Pinned remote sources are immutable, and a non-local path has no files to
        stat, so those keep the historical three-part key.
        """

        if remote_revision is None and remote_sha256 is None:
            generation = _local_adapter_generation(path)
            if generation is not None:
                return (path, None, None, generation)
        return (path, remote_revision, remote_sha256)

    def _load_adapter(self) -> Tuple[LoraConfig, Dict[str, torch.Tensor]]:
        """Load and cache the adapter config and weights for this source."""

        path, remote_revision, remote_sha256 = self._resolve_source()
        cache_key = self._adapter_cache_key(path, remote_revision, remote_sha256)
        lora_cache = AdapterCache.get(cache_key)
        if lora_cache is None:
            # get lora config
            lora_cfg = LoraConfig.from_pretrained(
                path=path,
                filename=HF_ADAPTER_CONFIG_FILE_NAME,
                allow_remote=remote_revision is not None,
                revision=remote_revision,
            )
            lora_cfg.gptqmodel_path = path  # hack: save this

            if not isinstance(lora_cfg, LoraConfig):
                raise ValueError(f"Adapter: Expected `LoraConfig` in `{path}`, actual = `{lora_cfg}`")

            lora_path = resolve_path(
                path,
                HF_ADAPTER_FILE_NAME,
                allow_remote=remote_revision is not None,
                revision=remote_revision,
                expected_sha256=remote_sha256,
            )

            # save to adapter cache
            AdapterCache.add(cache_key, lora_cfg, safetensors.torch.load_file(lora_path))

            lora_cache = AdapterCache.get(cache_key)
            assert lora_cache is not None

        lora_cfg, lora_weights = lora_cache
        _reject_unsupported_adapter_tensors(lora_weights)
        self._lora_config = lora_cfg
        return lora_cfg, lora_weights

    def adapter_config(self) -> Optional[LoraConfig]:
        """Return the PEFT config for this adapter, if it has one on disk.

        ``None`` covers adapters whose weights are merged into the model
        (EoRA) and output directories that have not been saved yet. Those keep
        the historical attach-to-every-module behavior.
        """

        if self._lora_config is not None:
            return self._lora_config
        if self.path in LORA_MERGED_WEIGHT_PATHS:
            return None
        path, remote_revision, _ = self._resolve_source()
        if not path or path in LORA_MERGED_WEIGHT_PATHS:
            return None
        if remote_revision is None:
            # A local EoRA output directory (or any adapter that has not been
            # saved yet) has no target list. Do not turn that into a hard error
            # while merely deciding which modules may attach the adapter.
            if not os.path.isdir(path):
                return None
            if not os.path.isfile(os.path.join(path, HF_ADAPTER_CONFIG_FILE_NAME)):
                return None
        lora_cfg, _ = self._load_adapter()
        return lora_cfg

    def targets_module(self, module_name: str) -> bool:
        """Whether this adapter's ``target_modules`` selects ``module_name``."""

        if self.lora_A is not None and self.lora_B is not None:
            # Preloaded merged A/B tensors carry no target list.
            return True
        lora_cfg = self.adapter_config()
        if lora_cfg is None:
            return True
        if module_matches_targets(
            module_name,
            lora_cfg.target_modules,
            lora_cfg.exclude_modules,
        ):
            return True
        # Legacy EoRA exports wrote the target list and tensor keys in lowercase
        # (`wqkv` for `Wqkv`). `has_weights` already accepts that spelling, but
        # this target check is evaluated first and rejected the module. Accept
        # the lowercase spelling only when the adapter's tensor pair confirms
        # it, so canonical PEFT files keep case-sensitive targeting (#5418).
        lowered = module_name.lower()
        if lowered == module_name:
            return False
        if not module_matches_targets(
            lowered,
            lora_cfg.target_modules,
            lora_cfg.exclude_modules,
        ):
            return False
        return self.has_weights(module_name)

    def has_weights(self, module_name: str) -> bool:
        """Whether the adapter source has both A and B tensors for a module."""

        if self.lora_A is not None and self.lora_B is not None:
            return True
        if self.adapter_config() is None:
            return True
        _, lora_weights = self._load_adapter()
        for key in (module_name, module_name.lower()):
            found_a = any(
                name.endswith(f"{key}.lora_A.weight") for name in lora_weights
            )
            found_b = any(
                name.endswith(f"{key}.lora_B.weight") for name in lora_weights
            )
            if found_a and found_b:
                return True
        return False

    def post_init(self, weight_key: str, device:torch.device, lora_A: torch.Tensor=None, lora_B: torch.Tensor=None):
        """Loads, caches, and materializes LoRA tensors for the target module."""

        # self.register_buffer("lora_A", lora_A)
        # self.register_buffer("lora_B", lora_B)

        # we need since lora A/B weights may be merged into model tensors and not separate
        if lora_A is not None and lora_B is not None:
            # print(f"Adapter has preloaded lora_A and lora_B")
            self.lora_A, self.lora_B = lora_A, lora_B
            return

        lora_cfg, lora_weights = self._load_adapter()
        cache_key = self._adapter_cache_key(
            self.path,
            getattr(self, "_gptqmodel_remote_revision", None),
            getattr(self, "_gptqmodel_remote_sha256", None),
        )
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
            # Record consumption instead of deleting the cached tensors. One
            # model load may legitimately use only a subset of the adapter's
            # target modules, and deleting the rest made the next load of the
            # same adapter fail because those keys were already gone.
            if AdapterCache.consume(cache_key, pop_keys):
                AdapterCache.remove(cache_key)
                log.info("Adapter: Consumed all Lora weights")

        else:
            log.warn(f"Adapter: Lora weights not found for `{weight_key}`")

        assert lora_A is not None and lora_B is not None, f"Adapter: `lora_A` and `lora_B` must both be present in the weights: actual = `{lora_A}` and `{lora_B}`"

        # check for rank override from base config
        self.dynamic_rank_override(lora_cfg=lora_cfg, weight_key=weight_key)
        alpha = self.dynamic_alpha_override(lora_cfg=lora_cfg, weight_key=weight_key)

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
        if self.rank <= 0:
            raise ValueError(f"Adapter: LoRA rank must be positive, actual = `{self.rank}`.")
        if alpha <= 0:
            raise ValueError(f"Adapter: LoRA alpha must be positive, actual = `{alpha}`.")
        self.lora_scale = (
            float(alpha) / float(self.rank) ** 0.5
            if lora_cfg.use_rslora
            else float(alpha) / float(self.rank)
        )
        # `lora_B` is already a private contiguous clone from the immutable
        # adapter cache. Fold the scale once here so every forward uses the
        # PEFT effective scale without adding another multiply, unless the fold
        # would not survive `lora_B`'s storage dtype (#5426).
        self._lora_scale_folded = self._fold_scale_into_b()

        #print(f"Adapter: lora_A {lora_A.shape}: `{lora_B}`")
        #print(f"Adapter: lora_B {lora_B.shape}: `{lora_B}`")

    def _fold_scale_into_b(self) -> bool:
        """Fold ``lora_scale`` into ``lora_B`` only when that is lossless enough.

        The fold saves one multiply per forward, but it runs in the adapter's
        storage dtype. PEFT multiplies the low-rank product by ``alpha / rank``
        after both projections, so a large ``B`` (or a large scale) can overflow
        FP16 to ``inf`` even though the scaled product with a small activation
        is finite (#5426). Folding can also flush a nonzero element to zero.

        Returns ``True`` when the scale was folded into ``lora_B``, ``False``
        when ``apply()`` must multiply the product instead.
        """

        scale = float(self.lora_scale)
        if self.lora_B is None or scale == 1.0:
            return True
        with torch.no_grad():
            folded = self.lora_B * scale
            if not bool(torch.isfinite(folded).all()):
                return False
            if bool(((folded == 0) & (self.lora_B != 0)).any()):
                return False
            self.lora_B.mul_(scale)
        return True

    def dynamic_rank_override(self, lora_cfg: LoraConfig, weight_key: str) -> bool:
        """Overrides the adapter rank when the config defines a matching rank pattern."""

        assert lora_cfg.rank_pattern is not None and weight_key is not None
        if lora_cfg.rank_pattern:
            for k, v in lora_cfg.rank_pattern.items():
                assert isinstance(k, str) and isinstance(v, int)
                assert v > 0 # check for invalid rank range
            # Exact case-sensitive matches first, then the legacy fallback; see
            # `_pattern_override_candidates` (#5422).
            for pattern, override in _pattern_override_candidates(
                lora_cfg.rank_pattern, weight_key
            ):
                self.rank = override
                log.info(f"Adapter: Base Lora `rank` = `{self.rank}` has been overridden by `{pattern}` due to dynamic `LoraConfig.rank_pattern` control.")
                return True

        return False

    def dynamic_alpha_override(self, lora_cfg: LoraConfig, weight_key: str) -> float:
        """Resolves the PEFT alpha for this module from `alpha_pattern` or the base config."""

        base_alpha = lora_cfg.lora_alpha
        alpha = base_alpha
        if lora_cfg.alpha_pattern:
            for pattern, override in lora_cfg.alpha_pattern.items():
                # PEFT accepts float alphas in `alpha_pattern`; JSON has no
                # integer type, so `4.0` is a valid export. Reject only values
                # that cannot be a positive scale.
                if not isinstance(pattern, str):
                    raise ValueError(
                        f"Adapter: `alpha_pattern` key must be a string, actual = `{pattern}`."
                    )
                if not isinstance(override, numbers.Real) or float(override) <= 0:
                    raise ValueError(
                        f"Adapter: `alpha_pattern[{pattern!r}]` must be a positive real, "
                        f"actual = `{override}`."
                    )
            # Exact case-sensitive matches first, then the legacy fallback; see
            # `_pattern_override_candidates` (#5422).
            for pattern, override in _pattern_override_candidates(
                lora_cfg.alpha_pattern, weight_key
            ):
                alpha = float(override)
                log.info(
                    f"Adapter: Base Lora `alpha` = `{base_alpha}` has been overridden by "
                    f"`{pattern}` due to dynamic `LoraConfig.alpha_pattern` control."
                )
                break

        return alpha



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
