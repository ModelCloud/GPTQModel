# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pure packing of a GPT-QModel GGUF checkpoint into one llama.cpp file.

Every quantized linear in a GPT-QModel GGUF checkpoint already stores ggml
block bytes in ``qweight``. This module only:

* renames tensors to llama.cpp's ``blk.N.*`` names,
* reorders or splits whole output rows where an architecture requires it,
* stores dense tensors (embeddings, norms, biases) at the target width,
* writes the architecture and tokenizer metadata llama.cpp reads.

It never dequantizes, re-quantizes, or rewrites a block. A layout that would
need a transform crossing a block boundary is refused before the output file is
created.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..utils.logger import setup_logger
from .arch import ArchSpec, head_dim, resolve_spec
from .config import GGUFPackageConfig, PackageConfig
from .errors import (
    CheckpointFormatError,
    MissingDependencyError,
    PackageError,
    UnsafeTransformError,
    UnsupportedArchitectureError,
    UnsupportedQuantTypeError,
    UnsupportedTensorError,
)
from .tokenizer import TokenizerMetadata, build_tokenizer_metadata, tokenizer_kind


log = setup_logger()

GGUF_QUANTIZATION_VERSION = 2

# These ggml types hold dot-product activations and have no weight dot kernel.
_ACTIVATION_ONLY_QTYPES = {"Q8_1", "Q8_K"}

#: Model types whose GGUF architecture name differs from ``model_type``.
MODEL_TYPE_TO_ARCH = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "llama": "llama",
}

#: Tensors llama.cpp may emit that carry no weights a GGUF file can store.
_SKIPPED_TENSOR_SUFFIXES = (
    ".rotary_emb.inv_freq",
    ".attention.masked_bias",
    ".attention.bias",
)

#: Leaves every dense-decoder block must provide.
_REQUIRED_BLOCK_LEAVES = (
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)

_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)

#: Hugging Face generation settings that llama.cpp reads back from the file.
#: ``general.sampling.sequence`` is deliberately not written: it would replace
#: llama.cpp's default sampler chain instead of only changing its parameters.
_SAMPLING_FIELDS = (
    ("temperature", "general.sampling.temp", "float"),
    ("top_k", "general.sampling.top_k", "int"),
    ("top_p", "general.sampling.top_p", "float"),
    ("min_p", "general.sampling.min_p", "float"),
    ("repetition_penalty", "general.sampling.penalty_repeat", "float"),
    ("mirostat", "general.sampling.mirostat", "int"),
    ("mirostat_tau", "general.sampling.mirostat_tau", "float"),
    ("mirostat_eta", "general.sampling.mirostat_eta", "float"),
)

_DENSE_ITEMSIZE = {"f16": 2, "bf16": 2, "f32": 4}
_SAFETENSORS_DTYPE_TO_DENSE = {
    "BF16": "bf16",
    "F16": "f16",
    "F32": "f32",
    "F64": "f32",
}
_DENSE_TO_GGML_TYPE = {"f16": "F16", "bf16": "BF16", "f32": "F32"}

_PLAN_ORDER = {
    "token_embd.weight": 0,
    "output_norm.weight": 1,
    "output.weight": 2,
}
_BLOCK_LEAF_ORDER = {
    "attn_norm.weight": 0,
    "attn_q.weight": 1,
    "attn_q.bias": 2,
    "attn_k.weight": 3,
    "attn_k.bias": 4,
    "attn_v.weight": 5,
    "attn_v.bias": 6,
    "attn_q_norm.weight": 7,
    "attn_k_norm.weight": 8,
    "attn_output.weight": 9,
    "ffn_norm.weight": 10,
    "ffn_gate.weight": 11,
    "ffn_up.weight": 12,
    "ffn_down.weight": 13,
}


@dataclass(frozen=True)
class _PlannedTensor:
    """One output tensor, decided without loading its payload."""

    name: str
    source: Optional[str]
    kind: str  # "packed" | "dense" | "generated"
    shape: Tuple[int, ...]
    qtype: str = ""
    dtype: str = ""
    storage_shape: Tuple[int, ...] = ()
    permutation: Optional[np.ndarray] = None
    generated: Optional[np.ndarray] = None


class _ShardReader:
    """Read tensors lazily from one or more safetensors shards."""

    def __init__(self, stack: ExitStack, handles: Mapping[str, Any], weight_map: Mapping[str, str]) -> None:
        self._stack = stack
        self._handles = dict(handles)
        self.weight_map = dict(weight_map)

    @classmethod
    def open(cls, model_dir: str) -> "_ShardReader":
        try:
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - safetensors is a base dependency
            raise MissingDependencyError("package(): reading a checkpoint needs `safetensors`.") from exc

        weight_map = _read_weight_map(model_dir)
        stack = ExitStack()
        handles: Dict[str, Any] = {}
        try:
            for shard in sorted(set(weight_map.values())):
                handles[shard] = stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
        except BaseException:
            stack.close()
            raise
        return cls(stack, handles, weight_map)

    def slice(self, name: str) -> Any:
        return self._handles[self.weight_map[name]].get_slice(name)

    def tensor(self, name: str) -> torch.Tensor:
        return self._handles[self.weight_map[name]].get_tensor(name)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "_ShardReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def package(config: PackageConfig) -> Path:
    """Write one container file from a GPT-QModel checkpoint.

    ``GGUFPackageConfig`` produces a single llama.cpp-readable ``.gguf`` file.
    The call fails closed: it refuses a checkpoint it cannot prove it packs
    byte-exactly, and it never publishes a partial or silently re-quantized
    file.
    """

    if not isinstance(config, PackageConfig):
        raise TypeError(f"package(): expected a PackageConfig instance, got {type(config).__name__}.")
    if isinstance(config, GGUFPackageConfig):
        return _package_gguf(config)
    raise TypeError(
        f"package(): no writer is registered for {type(config).__name__}; pass a GGUFPackageConfig."
    )


def _package_gguf(config: GGUFPackageConfig) -> Path:
    if sys.byteorder != "little":
        raise PackageError(
            "package(): GGUF output is written for little-endian hosts only; the packed rows come from a "
            "little-endian checkpoint and would need a byteswap on a big-endian host."
        )

    gguf = _gguf_module()
    model_dir = config.model_path
    model_config = _load_model_config(model_dir)
    quantize_config = _quantize_config(_load_quant_payload(model_dir, model_config))
    arch = _resolve_arch(model_config, config.architecture, gguf)
    spec = resolve_spec(arch, config.allow_unverified_arch)
    model_config = spec.with_config_defaults(model_config)

    with _ShardReader.open(model_dir) as reader:
        _refuse_moe(model_config, reader.weight_map, arch)
        n_blocks = _require_int(model_config, "num_hidden_layers")
        spec.validate_layout(model_config, n_blocks)
        tensor_map = gguf.get_tensor_name_map(_arch_enum(gguf, arch), n_blocks)
        tokenizer = _load_tokenizer(config, model_dir)
        tokenizer_meta = _build_tokenizer_metadata(config, tokenizer, model_config, spec)
        plan = _plan_tensors(
            reader=reader,
            tensor_map=tensor_map,
            quantize_config=quantize_config,
            spec=spec,
            model_config=model_config,
            config=config,
            n_blocks=n_blocks,
        )
        out = _resolve_output(config)
        _write_gguf(
            gguf=gguf,
            reader=reader,
            out=out,
            plan=plan,
            arch=arch,
            model_config=model_config,
            spec=spec,
            tokenizer_meta=tokenizer_meta,
            config=config,
            n_blocks=n_blocks,
            name=_model_name(config, model_config, model_dir),
            file_type=_file_type(gguf, quantize_config, config.file_type),
            generation_config=_load_generation_config(model_dir, model_config),
        )
    log.info("package: wrote %s (%d tensors)", out, len(plan))
    return Path(out)


def _gguf_module() -> Any:
    try:
        import gguf
    except Exception as exc:
        raise MissingDependencyError(
            "package(): GGUF output needs the optional `gguf` package. Install it with "
            "`pip install 'gptqmodel[gguf]'` (the extra pins gguf>=0.19.0)."
        ) from exc
    required = ("GGUFWriter", "GGMLQuantizationType", "MODEL_ARCH_NAMES", "get_tensor_name_map", "LlamaFileType")
    missing = [symbol for symbol in required if not hasattr(gguf, symbol)]
    if missing:
        raise MissingDependencyError(
            f"package(): the installed `gguf` package lacks {missing}. Install gguf>=0.19.0 with "
            "`pip install 'gptqmodel[gguf]'`."
        )
    return gguf


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_model_config(model_dir: str) -> Mapping[str, Any]:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise CheckpointFormatError(f"package(): `config.json` is missing from {model_dir!r}.")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CheckpointFormatError(f"package(): `config.json` in {model_dir!r} is not a JSON object.")
    return payload


def _load_quant_payload(model_dir: str, model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = model_config.get("quantization_config")
    if isinstance(payload, dict) and payload:
        return payload
    path = os.path.join(model_dir, "quantize_config.json")
    if os.path.isfile(path):
        payload = _load_json(path)
        if isinstance(payload, dict) and payload:
            return payload
    raise CheckpointFormatError(
        f"package(): no `quantization_config` in config.json and no quantize_config.json in {model_dir!r}. "
        "Only a GPT-QModel GGUF checkpoint can be packed."
    )


def _load_generation_config(model_dir: str, model_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the checkpoint's generation settings, or an empty mapping.

    ``generation_config.json`` is optional and is not part of the weight
    checkpoint, so a missing or malformed file is not an error: the model then
    simply carries no sampling defaults, exactly as llama.cpp behaves.
    """

    embedded = model_config.get("generation_config")
    if isinstance(embedded, dict) and embedded:
        return embedded
    path = os.path.join(model_dir, "generation_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        payload = _load_json(path)
    except (OSError, ValueError):
        log.warn("package: ignoring unreadable generation_config.json at %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _quantize_config(payload: Mapping[str, Any]) -> Any:
    from .._quant_methods import GGUF_EXPORT_QUANT_METHOD, LEGACY_GGUF_QUANT_METHOD
    from ..quantization.config import GGUFConfig, QuantizeConfig

    method = payload.get("quant_method")
    if method not in (GGUF_EXPORT_QUANT_METHOD, LEGACY_GGUF_QUANT_METHOD):
        raise CheckpointFormatError(
            "package(): expected a GPT-QModel GGUF checkpoint "
            f"(`quant_method: {GGUF_EXPORT_QUANT_METHOD!r}` or legacy "
            f"`quant_method: {LEGACY_GGUF_QUANT_METHOD!r}`), got {method!r}. This writer only packs "
            "GPT-QModel GGUF checkpoints; other quantization methods are not supported."
        )
    config = QuantizeConfig.from_quant_config(dict(payload))
    if not isinstance(config, GGUFConfig):
        raise CheckpointFormatError(
            f"package(): the checkpoint describes {type(config).__name__}, not a GGUF checkpoint."
        )
    return config


def _read_weight_map(model_dir: str) -> Dict[str, str]:
    from ..utils.checkpoint_paths import resolve_checkpoint_path

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        payload = _load_json(index_path)
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointFormatError(f"package(): {index_path!r} has no usable `weight_map`.")
        resolved: Dict[str, str] = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise CheckpointFormatError(f"package(): malformed weight_map entry {name!r}: {shard!r}.")
            resolved[name] = resolve_checkpoint_path(model_dir, shard, kind="safetensors shard")
        return resolved

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="np", device="cpu") as handle:
            names = list(handle.keys())
        if not names:
            raise CheckpointFormatError(f"package(): {single!r} contains no tensors.")
        return dict.fromkeys(names, single)

    raise CheckpointFormatError(
        f"package(): no `model.safetensors` or `model.safetensors.index.json` in {model_dir!r}. "
        "GPT-QModel writes safetensors checkpoints; a pickle checkpoint must be re-saved first."
    )


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise CheckpointFormatError(f"package(): `config.json` must declare an integer `{key}`, got {value!r}.")
    return value


def _resolve_arch(model_config: Mapping[str, Any], override: str, gguf: Any) -> str:
    arch = (override or "").strip().lower()
    if not arch:
        model_type = str(model_config.get("model_type") or "").strip().lower()
        arch = MODEL_TYPE_TO_ARCH.get(model_type, model_type)
    if not arch:
        raise UnsupportedArchitectureError(
            "package(): `config.json` declares no `model_type`; pass `architecture=`."
        )
    known = set(gguf.MODEL_ARCH_NAMES.values())
    if arch not in known:
        raise UnsupportedArchitectureError(
            f"package(): `{arch}` is not a llama.cpp architecture name. Pass `architecture=` with one of "
            f"the names the installed `gguf` package knows ({len(known)} entries)."
        )
    return arch


def _arch_enum(gguf: Any, arch: str) -> Any:
    for member, name in gguf.MODEL_ARCH_NAMES.items():
        if name == arch:
            return member
    raise UnsupportedArchitectureError(f"package(): the installed `gguf` package has no `{arch}` architecture.")


def _refuse_moe(model_config: Mapping[str, Any], weight_map: Mapping[str, str], arch: str) -> None:
    declared = [key for key in _MOE_CONFIG_KEYS if model_config.get(key)]
    expert_tensors = [name for name in weight_map if ".experts." in name or "_exps." in name]
    if not declared and not expert_tensors:
        return
    raise UnsupportedArchitectureError(
        f"package(): `{arch}` is a mixture-of-experts checkpoint "
        f"({', '.join(declared) or 'per-expert tensors'}; {len(expert_tensors)} expert tensor(s)). "
        "Packing routed experts needs an architecture-specific fusion this writer does not implement, so it "
        "is refused instead of writing a file llama.cpp cannot load."
    )


def _load_tokenizer(config: GGUFPackageConfig, model_dir: str) -> Any:
    if config.tokenizer is not None:
        return config.tokenizer

    # The packaging writer only needs the checkpoint's vocabulary, but class
    # selection and normalization stay owned by the package tokenizer loader:
    # importing ``transformers.AutoTokenizer`` here would pull a
    # ``transformers.models`` family into the packaging path, which the model
    # port policy forbids.
    from ..utils.hf import load_hf_tokenizer

    source = config.tokenizer_path or model_dir
    try:
        return load_hf_tokenizer(source, local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise CheckpointFormatError(
            f"package(): could not load a tokenizer from {source!r}: {exc}. Pass `tokenizer_path=` or a "
            "loaded `tokenizer=` object."
        ) from exc


def _build_tokenizer_metadata(
    config: GGUFPackageConfig,
    tokenizer: Any,
    model_config: Mapping[str, Any],
    spec: ArchSpec,
) -> TokenizerMetadata:
    kind = tokenizer_kind(tokenizer)
    pre = config.tokenizer_pre or spec.tokenizer_pre(kind)
    if pre is None:
        pre = "default"
        log.warn(
            "package: no pre-tokenizer is registered for architecture `%s` and tokenizer kind `%s`; "
            "writing `tokenizer.ggml.pre=default`. Set `tokenizer_pre=` if llama.cpp tokenizes differently.",
            spec.arch,
            kind,
        )
    return build_tokenizer_metadata(
        tokenizer,
        vocab_size=_require_int(model_config, "vocab_size"),
        tokenizer_pre=pre,
        chat_template=config.chat_template or None,
    )


def _is_skipped_tensor(name: str) -> bool:
    return ("." + name).endswith(_SKIPPED_TENSOR_SUFFIXES)


def _map_name(tensor_map: Any, hf_name: str) -> Optional[str]:
    return tensor_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))


def _module_alias(quantize_config: Any, module: str) -> str:
    from ..quantization.config import clone_weight_only_config_for_module

    return clone_weight_only_config_for_module(quantize_config, module).runtime_bits.to_string()


def _check_shape(gguf_name: str, shape: Tuple[int, ...], spec: ArchSpec, model_config: Mapping[str, Any]) -> None:
    expected = spec.expected_shape(gguf_name, model_config)
    if expected is None:
        return
    if tuple(int(value) for value in expected) == tuple(int(value) for value in shape):
        return
    raise UnsafeTransformError(
        f"package(): `{gguf_name}` has logical shape {tuple(shape)}, but the `{spec.arch}` layout requires "
        f"{tuple(expected)}. Packed rows are byte-exact only when the declared shape matches; padding, a "
        "column-wise permutation, or a fused layout would need a dequantize/requantize pass, which packaging "
        "does not perform."
    )


def _plan_dense_dtype(gguf_name: str, shape: Tuple[int, ...], source_dtype: str, choice: str) -> str:
    if len(shape) <= 1 or gguf_name.endswith("_norm.weight"):
        return "f32"
    if choice == "auto":
        return _SAFETENSORS_DTYPE_TO_DENSE.get(str(source_dtype).upper(), "f32")
    return choice


def _plan_tensors(
    *,
    reader: _ShardReader,
    tensor_map: Any,
    quantize_config: Any,
    spec: ArchSpec,
    model_config: Mapping[str, Any],
    config: GGUFPackageConfig,
    n_blocks: int,
) -> List[_PlannedTensor]:
    from ..nn_modules.qlinear.gguf import _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE, _GGUF_TYPE_INFO

    source_names = list(reader.weight_map)
    quantized_modules = {name[: -len(".qweight")] for name in source_names if name.endswith(".qweight")}

    planned: List[_PlannedTensor] = []
    unmapped: List[str] = []
    seen: Dict[str, str] = {}

    def record(tensor: _PlannedTensor, source: str) -> None:
        previous = seen.get(tensor.name)
        if previous is not None:
            raise UnsupportedTensorError(f"package(): `{source}` and `{previous}` both map to `{tensor.name}`.")
        seen[tensor.name] = source
        planned.append(tensor)

    for name in source_names:
        if _is_skipped_tensor(name):
            continue

        if name.endswith(".qweight"):
            module = name[: -len(".qweight")]
            gguf_name = _map_name(tensor_map, module + ".weight")
            if gguf_name is None:
                unmapped.append(name)
                continue
            alias = _module_alias(quantize_config, module)
            qtype = _GGUF_BITS_ALIAS_TO_TENSOR_QTYPE.get(alias)
            if qtype is None:
                raise UnsupportedQuantTypeError(
                    f"package(): `{module}` resolves to GGUF subtype `{alias}`, which has no ggml tensor type."
                )
            if qtype in _ACTIVATION_ONLY_QTYPES:
                raise UnsupportedQuantTypeError(
                    f"package(): `{module}` uses {qtype}, which llama.cpp cannot use as a weight type."
                )
            info = _GGUF_TYPE_INFO[qtype]
            packed = reader.slice(name)
            storage_dtype = packed.get_dtype()
            if storage_dtype != "U8":
                raise CheckpointFormatError(
                    f"package(): `{name}` must contain packed uint8 bytes, got {storage_dtype}."
                )
            storage_shape = tuple(int(value) for value in packed.get_shape())
            if len(storage_shape) != 2:
                raise CheckpointFormatError(
                    f"package(): `{name}` must be a 2-D packed matrix, got shape {storage_shape}."
                )
            if storage_shape[1] % info["type_size"] != 0:
                raise CheckpointFormatError(
                    f"package(): `{name}` row width {storage_shape[1]} is not a multiple of the "
                    f"{qtype} block size {info['type_size']}."
                )
            logical = (storage_shape[0], storage_shape[1] // info["type_size"] * info["block_size"])
            _check_shape(gguf_name, logical, spec, model_config)
            record(
                _PlannedTensor(
                    name=gguf_name,
                    source=name,
                    kind="packed",
                    shape=logical,
                    qtype=qtype,
                    storage_shape=storage_shape,
                    permutation=spec.row_permutation(gguf_name, model_config),
                ),
                name,
            )
            continue

        module, _, suffix = name.rpartition(".")
        if suffix in {"weight", "scales", "qzeros", "g_idx"} and module in quantized_modules:
            # A GGUF module owns only `qweight` (and an optional dense `bias`).
            raise CheckpointFormatError(
                f"package(): `{module}` is a packed GGUF module but the checkpoint also stores `{name}`. "
                "A GGUF checkpoint stores only `qweight` and an optional `bias` for a quantized linear."
            )

        gguf_name = _map_name(tensor_map, name)
        if gguf_name is None:
            unmapped.append(name)
            continue
        slice_ = reader.slice(name)
        shape = tuple(int(value) for value in slice_.get_shape())
        _check_shape(gguf_name, shape, spec, model_config)
        record(
            _PlannedTensor(
                name=gguf_name,
                source=name,
                kind="dense",
                shape=shape,
                dtype=_plan_dense_dtype(gguf_name, shape, slice_.get_dtype(), config.dense_dtype),
                permutation=spec.row_permutation(gguf_name, model_config),
            ),
            name,
        )

    for gguf_name, array in spec.extra_tensors(model_config).items():
        record(
            _PlannedTensor(
                name=gguf_name,
                source=None,
                kind="generated",
                shape=tuple(int(value) for value in array.shape),
                generated=array,
            ),
            "<generated>",
        )

    if unmapped:
        raise UnsupportedTensorError(
            "package(): these checkpoint tensors have no llama.cpp name for "
            f"`{spec.arch}`: {sorted(unmapped)}. Refusing to drop weights silently."
        )

    missing = _missing_required(planned, n_blocks, spec, model_config)
    if missing:
        raise CheckpointFormatError(
            f"package(): the checkpoint is missing tensors a `{spec.arch}` GGUF file needs: {missing}."
        )

    # The verified model configs default to untied embeddings.
    if model_config.get("tie_word_embeddings", not spec.verified) is False and "output.weight" not in seen:
        raise CheckpointFormatError(
            "package(): `tie_word_embeddings` is false but the checkpoint has no `lm_head`; llama.cpp would "
            "tie the output projection to the embedding and change the model."
        )

    return sorted(planned, key=_plan_sort_key)


def _missing_required(
    planned: Sequence[_PlannedTensor],
    n_blocks: int,
    spec: ArchSpec,
    model_config: Mapping[str, Any],
) -> List[str]:
    present = {tensor.name for tensor in planned}
    required = {"token_embd.weight", "output_norm.weight"}
    if spec.verified:
        leaves = _REQUIRED_BLOCK_LEAVES + spec.required_leaves(model_config)
        for block in range(n_blocks):
            required.update(f"blk.{block}.{leaf}" for leaf in leaves)
    return sorted(required - present)


def _plan_sort_key(tensor: _PlannedTensor) -> Tuple[int, int, int, str]:
    direct = _PLAN_ORDER.get(tensor.name)
    if direct is not None:
        return (0, direct, 0, tensor.name)
    parts = tensor.name.split(".")
    if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
        leaf = ".".join(parts[2:])
        return (1, int(parts[1]), _BLOCK_LEAF_ORDER.get(leaf, 99), tensor.name)
    return (2, 0, 0, tensor.name)


def _model_name(config: GGUFPackageConfig, model_config: Mapping[str, Any], model_dir: str) -> str:
    if config.name:
        return config.name
    source = model_config.get("_name_or_path")
    if isinstance(source, str) and source.strip():
        return os.path.basename(os.path.normpath(source)) or source.strip()
    return os.path.basename(os.path.normpath(model_dir)) or "model"


def _file_type(gguf: Any, quantize_config: Any, override: int) -> int:
    if override >= 0:
        return override
    alias = quantize_config.runtime_bits.to_string()
    member = getattr(gguf.LlamaFileType, f"MOSTLY_{alias.upper()}", None)
    return int(member) if member is not None else -1


def _resolve_output(config: GGUFPackageConfig) -> str:
    out = config.out_path
    if os.path.lexists(out) and not config.overwrite:
        raise FileExistsError(f"package(): {out!r} already exists; pass `overwrite=True` to replace it.")
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return out


def _write_gguf(
    *,
    gguf: Any,
    reader: _ShardReader,
    out: str,
    plan: Sequence[_PlannedTensor],
    arch: str,
    model_config: Mapping[str, Any],
    spec: ArchSpec,
    tokenizer_meta: TokenizerMetadata,
    config: GGUFPackageConfig,
    n_blocks: int,
    name: str,
    file_type: int,
    generation_config: Mapping[str, Any],
) -> None:
    parent = os.path.dirname(out) or "."
    handle, temp_name = tempfile.mkstemp(prefix=".gguf-package-", suffix=".gguf", dir=parent)
    temp_path = Path(temp_name)
    try:
        os.close(handle)
        writer = gguf.GGUFWriter(path=None, arch=arch, use_temp_file=False)
        try:
            _add_metadata(
                writer=writer,
                model_config=model_config,
                spec=spec,
                tokenizer_meta=tokenizer_meta,
                config=config,
                n_blocks=n_blocks,
                name=name,
                file_type=file_type,
                generation_config=generation_config,
            )
            for tensor in plan:
                _add_tensor_info(writer, gguf, tensor)
            writer.write_header_to_file(path=temp_path)
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for tensor in plan:
                writer.write_tensor_data(_materialize(reader, tensor))
            writer.close()
        except BaseException:
            # Keep the write error when closing an already failing writer.
            with suppress(Exception):
                writer.close()
            raise

        _verify_written(temp_path, plan)
        if config.overwrite:
            os.replace(temp_path, out)
        else:
            # Publish the complete file only if its destination is still free.
            os.link(temp_path, out)
        # ``mkstemp`` creates the temporary file with 0600; a published
        # checkpoint has to be readable by the processes that serve it.
        os.chmod(out, 0o644)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warn("package: could not remove temporary file %s: %s", temp_path, exc)


def _add_metadata(
    *,
    writer: Any,
    model_config: Mapping[str, Any],
    spec: ArchSpec,
    tokenizer_meta: TokenizerMetadata,
    config: GGUFPackageConfig,
    n_blocks: int,
    name: str,
    file_type: int,
    generation_config: Mapping[str, Any],
) -> None:
    writer.add_name(name)
    if file_type >= 0:
        writer.add_file_type(file_type)
    writer.add_quantization_version(GGUF_QUANTIZATION_VERSION)
    writer.add_custom_alignment(config.alignment)

    writer.add_block_count(n_blocks)
    _maybe_uint32(writer, "add_context_length", model_config.get("max_position_embeddings"))
    writer.add_embedding_length(_require_int(model_config, "hidden_size"))
    _maybe_uint32(writer, "add_feed_forward_length", model_config.get("intermediate_size"))
    writer.add_head_count(_require_int(model_config, "num_attention_heads"))
    _maybe_uint32(writer, "add_head_count_kv", model_config.get("num_key_value_heads"))

    dim = head_dim(model_config)
    writer.add_rope_dimension_count(dim)
    writer.add_key_length(dim)
    writer.add_value_length(dim)
    _maybe_float32(writer, "add_layer_norm_rms_eps", model_config.get("rms_norm_eps"))
    _maybe_float32(writer, "add_layer_norm_eps", model_config.get("layer_norm_eps"))
    writer.add_vocab_size(_require_int(model_config, "vocab_size"))

    for key, value in spec.extra_metadata(model_config).items():
        _add_scalar(writer, key, value)

    writer.add_tokenizer_model(tokenizer_meta.model_kind)
    writer.add_tokenizer_pre(tokenizer_meta.pre)
    writer.add_token_list(list(tokenizer_meta.tokens))
    writer.add_token_types(list(tokenizer_meta.token_types))
    if tokenizer_meta.merges:
        writer.add_token_merges(list(tokenizer_meta.merges))
    for kind, token_id in tokenizer_meta.special_ids.items():
        getattr(writer, f"add_{kind}_token_id")(token_id)
    for kind, value in tokenizer_meta.add_special.items():
        getattr(writer, f"add_add_{kind}_token")(value)
    if tokenizer_meta.chat_template:
        writer.add_chat_template(tokenizer_meta.chat_template)

    _add_sampling_metadata(writer, generation_config, config.metadata)

    # Sampling defaults are caller-tunable; every other generated key is not.
    sampling_overrides = {
        key: value for key, value in config.metadata.items() if key.startswith("general.sampling.")
    }
    structural = {key: value for key, value in config.metadata.items() if key not in sampling_overrides}
    duplicates = structural.keys() & writer.kv_data[0].keys()
    if duplicates:
        raise PackageError(f"package(): metadata cannot replace generated keys: {sorted(duplicates)}.")
    for key, value in config.metadata.items():
        _add_scalar(writer, key, value)


def _add_sampling_metadata(
    writer: Any,
    generation_config: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> None:
    """Store Hugging Face generation settings under llama.cpp's keys.

    llama.cpp reads these as *parameters* for its default sampler chain, so
    they change defaults without replacing the chain. A caller-supplied
    ``metadata`` entry for the same key wins.

    ``do_sample: false`` means greedy decoding in Transformers, where
    ``temperature`` is ignored; the faithful mapping is ``temp = 0``.
    """

    do_sample = generation_config.get("do_sample")
    for source_key, gguf_key, kind in _SAMPLING_FIELDS:
        if gguf_key in overrides:
            continue
        value = generation_config.get(source_key)
        if source_key == "temperature" and do_sample is False:
            value = 0.0
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CheckpointFormatError(
                f"package(): `generation_config.{source_key}` must be a number, got {value!r}."
            )
        if kind == "int":
            if int(value) != value:
                raise CheckpointFormatError(
                    f"package(): `generation_config.{source_key}` must be an integer, got {value!r}."
                )
            writer.add_int32(gguf_key, int(value))
        else:
            writer.add_float32(gguf_key, float(value))


def _maybe_uint32(writer: Any, method: str, value: Any) -> None:
    if value is None:
        return
    if type(value) is not int or value < 0:
        raise CheckpointFormatError(f"package(): `{method[4:]}` must be a non-negative integer, got {value!r}.")
    getattr(writer, method)(value)


def _maybe_float32(writer: Any, method: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CheckpointFormatError(f"package(): `{method[4:]}` must be a number, got {value!r}.")
    getattr(writer, method)(float(value))


def _add_scalar(writer: Any, key: str, value: Any) -> None:
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        if -(2**63) <= value < 0:
            writer.add_int64(key, value)
        elif 0 <= value < 2**32:
            writer.add_uint32(key, value)
        elif 0 <= value < 2**64:
            writer.add_uint64(key, value)
        else:
            raise PackageError(f"package(): metadata integer `{key}` is outside the GGUF 64-bit range.")
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        raise PackageError(f"package(): metadata `{key}` has unsupported type {type(value).__name__}.")


def _add_tensor_info(writer: Any, gguf: Any, tensor: _PlannedTensor) -> None:
    if tensor.kind == "packed":
        writer.add_tensor_info(
            tensor.name,
            tensor.storage_shape,
            np.uint8,
            int(np.prod(tensor.storage_shape)),
            raw_dtype=_qtype_enum(gguf, tensor.qtype),
        )
        return
    if tensor.kind == "generated":
        array = tensor.generated
        assert array is not None
        writer.add_tensor_info(tensor.name, tuple(array.shape), array.dtype, int(array.nbytes))
        return
    if tensor.dtype == "bf16":
        byte_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
        writer.add_tensor_info(
            tensor.name,
            byte_shape,
            np.uint8,
            int(np.prod(tensor.shape)) * _DENSE_ITEMSIZE["bf16"],
            raw_dtype=_qtype_enum(gguf, "BF16"),
        )
        return
    np_dtype = np.float16 if tensor.dtype == "f16" else np.float32
    writer.add_tensor_info(
        tensor.name,
        tensor.shape,
        np.dtype(np_dtype),
        int(np.prod(tensor.shape)) * _DENSE_ITEMSIZE[tensor.dtype],
    )


def _qtype_enum(gguf: Any, name: str) -> Any:
    member = getattr(gguf.GGMLQuantizationType, name, None)
    if member is None:
        raise UnsupportedQuantTypeError(
            f"package(): the installed `gguf` package has no GGML tensor type `{name}`. Install gguf>=0.19.0 "
            "with `pip install 'gptqmodel[gguf]'`."
        )
    return member


def _materialize(reader: _ShardReader, tensor: _PlannedTensor) -> np.ndarray:
    if tensor.kind == "generated":
        assert tensor.generated is not None
        return np.ascontiguousarray(tensor.generated)

    assert tensor.source is not None
    payload = reader.tensor(tensor.source)
    if tensor.kind == "packed":
        array = payload.numpy()
        if tensor.permutation is not None:
            array = array[tensor.permutation]
        return np.ascontiguousarray(array)

    if tensor.permutation is not None:
        order = torch.as_tensor(tensor.permutation, dtype=torch.long)
        payload = payload.index_select(0, order)
    if tensor.dtype == "bf16":
        if payload.dtype != torch.bfloat16:
            payload = payload.to(torch.bfloat16)
        raw = payload.contiguous().view(torch.int16).numpy().view(np.uint8)
        return np.ascontiguousarray(raw)
    if tensor.dtype == "f16":
        return np.ascontiguousarray(payload.to(torch.float16).numpy())
    return np.ascontiguousarray(payload.to(torch.float32).numpy())


def _verify_written(path: Path, plan: Sequence[_PlannedTensor]) -> None:
    from ..utils.internal_gguf import GGUFReader

    reader = GGUFReader(path)
    written: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for tensor in reader.tensors:
        logical = tuple(int(value) for value in reversed(tensor.shape.tolist()))
        written[tensor.name] = (tensor.tensor_type.name, logical)

    expected = {tensor.name: (_ggml_type_name(tensor), tensor.shape) for tensor in plan}
    if written != expected:
        missing = sorted(set(expected) - set(written))
        extra = sorted(set(written) - set(expected))
        mismatched = sorted(name for name in set(expected) & set(written) if written[name] != expected[name])
        raise PackageError(
            "package(): the written file failed its own read-back check "
            f"(missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]})."
        )


def _ggml_type_name(tensor: _PlannedTensor) -> str:
    if tensor.kind == "packed":
        return tensor.qtype
    if tensor.kind == "generated":
        return "F32"
    return _DENSE_TO_GGML_TYPE[tensor.dtype]


__all__ = ["GGUF_QUANTIZATION_VERSION", "MODEL_TYPE_TO_ARCH", "package"]
