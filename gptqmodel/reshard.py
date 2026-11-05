# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Re-layout a safetensors checkpoint without loading any model weights.

``reshard`` copies tensor payloads byte-for-byte from the source checkpoint
into a new shard layout. Quantized payloads (``qweight``/``qzeros``/``scales``/
``g_idx`` and packed formats) are never dequantized, expanded, or materialized
in memory, so the operation is safe for very large and quantized checkpoints.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

from safetensors import safe_open

from .models._const import DEFAULT_MAX_SHARD_SIZE
from .utils.logger import setup_logger


log = setup_logger()

_LAYOUT_ALIASES = {
    "single": "single",
    "single_file": "single",
    "none": "single",
    "per_layer": "per_layer",
    "layer": "per_layer",
    "per_layer_with_single_global": "per_layer_with_single_global",
    "layer_with_single_global": "per_layer_with_single_global",
    "per_layer_single_global": "per_layer_with_single_global",
}

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")
_SOURCE_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.safetensors.index.json")
_MAX_SHARD_SIZE_RE = re.compile(r"\s*(\d+)([KMGTP]?B?)\s*", re.IGNORECASE)
_INDEXED_TENSOR_RE = re.compile(r"^(?P<prefix>.+?)\.(?P<index>\d+)\.")
_STREAM_CHUNK_SIZE = 32 * 1024 * 1024


class ShardLayout(str, Enum):
    """Output shard layout for :func:`reshard`."""

    #: One safetensors file for the entire checkpoint.
    SINGLE = "single"
    #: One directory per transformer layer; non-layer tensors are sharded at the root.
    PER_LAYER = "per_layer"
    #: One directory per transformer layer plus exactly one root file holding every
    #: non-layer tensor (embeddings, lm-head, norms, and other global modules).
    PER_LAYER_WITH_SINGLE_GLOBAL = "per_layer_with_single_global"

    @classmethod
    def coerce(cls, value: Union["ShardLayout", str]) -> "ShardLayout":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"layout must be a ShardLayout or str, got {type(value).__name__}")
        key = re.sub(r"[\s\-]+", "_", value.strip().lower())
        normalized = _LAYOUT_ALIASES.get(key)
        if normalized is None:
            supported = ", ".join(layout.value for layout in cls)
            raise ValueError(f"Unsupported shard layout: {value!r}. Supported layouts: {supported}.")
        return cls(normalized)


def _parse_max_shard_size(value: Optional[Union[int, str]]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_shard_size must be a positive integer or size string.")
        return value
    if not isinstance(value, str):
        raise TypeError("max_shard_size must be an int, a size string, or None.")
    match = _MAX_SHARD_SIZE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid max_shard_size value: {value}")
    base = int(match.group(1))
    suffix = match.group(2).upper()
    multiplier = 1
    if suffix.startswith("K"):
        multiplier = 1024
    elif suffix.startswith("M"):
        multiplier = 1024**2
    elif suffix.startswith("G"):
        multiplier = 1024**3
    elif suffix.startswith("T"):
        multiplier = 1024**4
    elif suffix.startswith("P"):
        multiplier = 1024**5
    if base <= 0:
        raise ValueError("max_shard_size must be positive.")
    return base * multiplier


@dataclass
class ShardConfig:
    """Shard layout and per-group size limits for :func:`reshard`.

    ``max_shard_size`` is applied inside each layer directory. In
    :attr:`ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL` the root global file is
    always written as exactly one shard, regardless of this limit.
    """

    layout: Union[ShardLayout, str] = ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL
    max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE
    #: Optional explicit layer prefixes such as ``["model.layers"]``. When omitted,
    #: GPT-QModel derives them from the checkpoint config or detects them from names.
    layer_prefixes: Optional[Sequence[str]] = None

    def __post_init__(self):
        self.layout = ShardLayout.coerce(self.layout)
        self.max_shard_size = _parse_max_shard_size(self.max_shard_size)
        if self.layer_prefixes is not None:
            if isinstance(self.layer_prefixes, str):
                self.layer_prefixes = [self.layer_prefixes]
            prefixes = tuple(str(prefix).strip(".") for prefix in self.layer_prefixes if str(prefix).strip("."))
            self.layer_prefixes = prefixes or None


@dataclass(frozen=True)
class _TensorRef:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    source_path: str
    data_offset: int
    num_bytes: int


def _read_safetensors_header(path: str) -> Tuple[Dict[str, object], int]:
    with open(path, "rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Invalid safetensors file (truncated header): {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"Invalid safetensors file (truncated header): {path}")
        data_start = 8 + header_size
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return header, data_start


def _resolve_source_dir(from_model) -> str:
    if isinstance(from_model, (str, os.PathLike)):
        path = os.fspath(from_model)
        if os.path.isdir(path):
            return path
        if os.path.isfile(path) and path.endswith((".safetensors", ".safetensors.index.json")):
            return os.path.dirname(os.path.abspath(path))
        raise FileNotFoundError(f"`from_model` is not a checkpoint directory: {path!r}")

    local_path = getattr(from_model, "model_local_path", None)
    if local_path is not None and os.path.isdir(os.fspath(local_path)):
        return os.fspath(local_path)
    raise TypeError(
        "`from_model` must be a checkpoint directory/path or a loaded GPTQModel with a local `model_local_path`."
    )


def _load_weight_map(source_dir: str) -> Dict[str, str]:
    for index_name in _SOURCE_INDEX_NAMES:
        index_path = os.path.join(source_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return {str(name): str(filename) for name, filename in weight_map.items()}

    candidates = sorted(name for name in os.listdir(source_dir) if name.endswith(".safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No safetensors checkpoint found in {source_dir!r}. "
            "Resharding requires safetensors input; convert pickle weights first."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple safetensors files without an index in {source_dir!r}: {candidates}. "
            "Provide a `model.safetensors.index.json`."
        )
    shard_name = candidates[0]
    with safe_open(os.path.join(source_dir, shard_name), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if not keys:
        raise ValueError(f"No tensors found in safetensors file: {shard_name}")
    return {str(name): shard_name for name in keys}


def _load_tensor_refs(
    source_dir: str,
    weight_map: Dict[str, str],
) -> Tuple[Dict[str, _TensorRef], Dict[str, str]]:
    refs: Dict[str, _TensorRef] = {}
    header_cache: Dict[str, Tuple[Dict[str, object], int]] = {}

    for name in sorted(weight_map):
        shard = weight_map[name]
        source_path = os.path.join(source_dir, shard)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Checkpoint shard listed in the index is missing: {source_path}")
        if source_path not in header_cache:
            header_cache[source_path] = _read_safetensors_header(source_path)
        header, data_start = header_cache[source_path]
        info = header.get(name)
        if info is None:
            raise ValueError(f"Tensor `{name}` listed in the index is missing from `{source_path}`")
        try:
            dtype = str(info["dtype"])
            shape = tuple(int(dim) for dim in info["shape"])
            start, end = (int(info["data_offsets"][0]), int(info["data_offsets"][1]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid safetensors metadata for `{name}` in `{source_path}`") from exc
        refs[name] = _TensorRef(
            name=name,
            dtype=dtype,
            shape=shape,
            source_path=source_path,
            data_offset=data_start + start,
            num_bytes=end - start,
        )

    known = set(weight_map)
    metadata: Dict[str, str] = {}
    for source_path in sorted(header_cache):
        header, _ = header_cache[source_path]
        extra = sorted(key for key in header if key != "__metadata__" and key not in known)
        if extra:
            raise ValueError(
                f"`{source_path}` contains tensors missing from the index, e.g. {extra[:3]}. "
                "Refusing to drop them during reshard."
            )
        shard_metadata = header.get("__metadata__") or {}
        if not isinstance(shard_metadata, dict):
            continue
        for key, value in shard_metadata.items():
            key = str(key)
            value = str(value)
            if key in metadata and metadata[key] != value:
                log.warning(
                    "reshard: conflicting safetensors metadata for `%s`; keeping %r",
                    key,
                    metadata[key],
                )
                continue
            metadata[key] = value

    metadata.setdefault("format", "pt")
    return refs, metadata


def _detect_layer_prefixes(names: Sequence[str]) -> List[str]:
    counts = Counter()
    indexes = defaultdict(set)
    for name in names:
        match = _INDEXED_TENSOR_RE.match(name)
        if match is None:
            continue
        prefix = match.group("prefix")
        counts[prefix] += 1
        indexes[prefix].add(int(match.group("index")))

    multi_layer = [prefix for prefix, values in indexes.items() if len(values) >= 2]
    if multi_layer:
        return sorted(multi_layer, key=lambda prefix: (-counts[prefix], prefix))
    if indexes:
        return [max(indexes, key=lambda prefix: (counts[prefix], prefix))]
    return []


def _resolve_layer_prefixes(
    from_model,
    source_dir: str,
    config: ShardConfig,
    trust_remote_code: bool,
    names: Sequence[str],
) -> List[str]:
    if config.layer_prefixes:
        return list(config.layer_prefixes)

    extractor = getattr(from_model, "extract_layers_node", None)
    if callable(extractor):
        try:
            prefixes = [str(prefix) for prefix in extractor() if prefix]
        except Exception:
            prefixes = []
        if prefixes:
            return prefixes

    try:
        from .models.auto import check_and_get_model_definition

        model_class = check_and_get_model_definition(source_dir, trust_remote_code=trust_remote_code)
        prefixes = [str(prefix) for prefix in model_class.extract_layers_node() if prefix]
        if prefixes:
            return prefixes
    except Exception:
        log.debug("reshard: could not derive layer prefixes from the checkpoint config", exc_info=True)

    return _detect_layer_prefixes(names)


def _resolve_layer_group(name: str, prefixes: Sequence[str]) -> Optional[str]:
    for prefix in sorted(prefixes, key=len, reverse=True):
        marker = f"{prefix}."
        if not name.startswith(marker):
            continue
        index, dot, _ = name[len(marker) :].partition(".")
        if dot and index.isdigit():
            return f"{prefix}.{index}"
    return None


def _layer_group_sort_key(group: str) -> Tuple[str, int]:
    prefix, _, index = group.rpartition(".")
    if index.isdigit():
        return prefix, int(index)
    return group, -1


def _group_tensors(
    refs: Dict[str, _TensorRef],
    prefixes: Sequence[str],
    layout: ShardLayout,
) -> List[Tuple[str, List[_TensorRef]]]:
    if layout is ShardLayout.SINGLE:
        return [("", [refs[name] for name in sorted(refs)])]

    groups: Dict[str, List[_TensorRef]] = defaultdict(list)
    for name in sorted(refs):
        groups[_resolve_layer_group(name, prefixes) or ""].append(refs[name])

    ordered = [(group, groups[group]) for group in sorted((g for g in groups if g), key=_layer_group_sort_key)]
    if "" in groups:
        ordered.append(("", groups[""]))
    return ordered


def _plan_shards(entries: List[_TensorRef], max_shard_size: Optional[int]) -> List[List[_TensorRef]]:
    if not max_shard_size or max_shard_size <= 0:
        return [entries]

    shards: List[List[_TensorRef]] = []
    current: List[_TensorRef] = []
    current_size = 0

    for entry in entries:
        if entry.num_bytes > max_shard_size:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([entry])
            continue
        if current and current_size + entry.num_bytes > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(entry)
        current_size += entry.num_bytes

    if current:
        shards.append(current)
    return shards


def _copy_range(source_path: str, target_handle, offset: int, length: int) -> None:
    remaining = length
    with open(source_path, "rb", buffering=0) as source:
        source.seek(offset)
        while remaining > 0:
            chunk = source.read(min(_STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                raise OSError(f"Unexpected EOF while copying from {source_path}")
            target_handle.write(chunk)
            remaining -= len(chunk)


def _write_safetensors(path: str, entries: List[_TensorRef], metadata: Dict[str, str]) -> int:
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata

    offset = 0
    for entry in entries:
        header[entry.name] = {
            "dtype": entry.dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + entry.num_bytes],
        }
        offset += entry.num_bytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Safetensors pads the JSON header to an 8-byte boundary.
    header_bytes += b" " * ((-len(header_bytes)) % 8)

    with open(path, "wb") as target:
        target.write(struct.pack("<Q", len(header_bytes)))
        target.write(header_bytes)
        for entry in entries:
            _copy_range(entry.source_path, target, entry.data_offset, entry.num_bytes)

    return os.path.getsize(path)


def _write_group(
    entries: List[_TensorRef],
    directory: str,
    base_name: str,
    single_file_name: str,
    metadata: Dict[str, str],
    max_shard_size: Optional[int],
) -> Tuple[List[str], Dict[str, str], int]:
    os.makedirs(directory, exist_ok=True)
    shards = _plan_shards(entries, max_shard_size)
    num_shards = len(shards)

    filenames: List[str] = []
    tensor_to_filename: Dict[str, str] = {}
    total_size = 0

    for index, shard in enumerate(shards, start=1):
        if num_shards == 1:
            filename = single_file_name
        else:
            filename = f"{base_name}-{index:05d}-of-{num_shards:05d}.safetensors"
        total_size += _write_safetensors(os.path.join(directory, filename), shard, metadata)
        filenames.append(filename)
        for entry in shard:
            tensor_to_filename[entry.name] = filename

    return filenames, tensor_to_filename, total_size


def _copy_auxiliary_files(source_dir: str, staging_dir: str) -> None:
    skipped_dirs: List[str] = []
    for entry in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, entry)
        if os.path.isdir(source_path):
            skipped_dirs.append(entry)
            continue
        lower = entry.lower()
        if lower.endswith(_WEIGHT_SUFFIXES) or lower.endswith(_INDEX_SUFFIXES):
            continue
        shutil.copy2(source_path, os.path.join(staging_dir, entry))

    if skipped_dirs:
        log.warning(
            "reshard: skipped source subdirectories (weight shards are rewritten): %s",
            ", ".join(sorted(skipped_dirs)),
        )


def _validate_destination(target_dir: str, overwrite: bool) -> None:
    if not os.path.exists(target_dir):
        return
    if not os.path.isdir(target_dir):
        raise NotADirectoryError(f"`to_model_path` exists and is not a directory: {target_dir}")
    if os.listdir(target_dir) and not overwrite:
        raise FileExistsError(
            f"`to_model_path` already exists and is not empty: {target_dir}. Pass overwrite=True to replace it."
        )


def reshard(
    from_model,
    to_model_path: Union[str, os.PathLike],
    config: Optional[ShardConfig] = None,
    *,
    shard_config: Optional[ShardConfig] = None,
    overwrite: bool = False,
    trust_remote_code: bool = False,
) -> str:
    """Rewrite an existing checkpoint into a new shard layout.

    Args:
        from_model: Source checkpoint directory, a ``model.safetensors`` or
            ``model.safetensors.index.json`` path, or a loaded ``GPTQModel``
            whose ``model_local_path`` points at the checkpoint.
        to_model_path: Destination directory. Weights are copied byte-for-byte;
            config, tokenizer, and other non-weight files are copied alongside.
        config: :class:`ShardConfig`. Defaults to
            ``ShardConfig(layout=ShardLayout.PER_LAYER_WITH_SINGLE_GLOBAL)``.
        shard_config: Alias for ``config``.
        overwrite: Replace an existing non-empty destination directory.
        trust_remote_code: Allow remote code while reading ``config.json`` to
            derive layer prefixes. Name-based detection is used when disabled.

    Returns:
        The destination directory path.

    The default layout writes one directory per transformer layer
    (``model.layers.0/layer.safetensors``) and exactly one root
    ``model.safetensors`` containing embeddings, lm-head, norms, and every other
    non-layer tensor.
    """
    if shard_config is not None:
        if config is not None:
            raise TypeError("Pass either `config` or `shard_config`, not both.")
        config = shard_config
    if config is None:
        config = ShardConfig()
    elif not isinstance(config, ShardConfig):
        if isinstance(config, (str, ShardLayout)):
            config = ShardConfig(layout=config)
        else:
            raise TypeError("`config` must be a ShardConfig instance.")

    source_dir = _resolve_source_dir(from_model)
    target_dir = os.fspath(to_model_path)
    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        raise ValueError("`to_model_path` must differ from the source checkpoint directory.")
    _validate_destination(target_dir, overwrite)

    weight_map = _load_weight_map(source_dir)
    refs, metadata = _load_tensor_refs(source_dir, weight_map)
    if not refs:
        raise ValueError(f"No tensors found in checkpoint: {source_dir}")

    prefixes = _resolve_layer_prefixes(
        from_model,
        source_dir,
        config,
        trust_remote_code,
        list(refs),
    )
    if config.layout is not ShardLayout.SINGLE and not prefixes:
        raise ValueError(
            "Unable to detect transformer layer prefixes. Pass "
            "ShardConfig(layer_prefixes=['model.layers']) explicitly."
        )

    groups = _group_tensors(refs, prefixes, config.layout)

    parent_dir = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".gptqmodel-reshard-", dir=parent_dir)
    backup_dir: Optional[str] = None

    try:
        _copy_auxiliary_files(source_dir, staging_dir)

        total_size = 0
        tensor_to_filename: Dict[str, str] = {}

        for group_name, entries in groups:
            if not group_name:
                directory = staging_dir
                base_name = "model"
                single_file_name = "model.safetensors"
                max_shard_size = config.max_shard_size if config.layout is ShardLayout.PER_LAYER else None
            else:
                directory = os.path.join(staging_dir, group_name)
                base_name = "layer"
                single_file_name = "layer.safetensors"
                max_shard_size = config.max_shard_size

            _, group_map, group_size = _write_group(
                entries,
                directory,
                base_name,
                single_file_name,
                metadata,
                max_shard_size,
            )
            total_size += group_size
            relative_prefix = f"{group_name}/" if group_name else ""
            for name, filename in group_map.items():
                tensor_to_filename[name] = f"{relative_prefix}{filename}"

        if set(tensor_to_filename) != set(refs):
            missing = sorted(set(refs) - set(tensor_to_filename))[:3]
            raise RuntimeError(f"Internal reshard error: tensors were not written: {missing}")

        unique_files = set(tensor_to_filename.values())
        if len(unique_files) > 1 or any("/" in filename for filename in unique_files):
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": tensor_to_filename,
            }
            index_path = os.path.join(staging_dir, "model.safetensors.index.json")
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2, sort_keys=True)
                handle.write("\n")

        if os.path.exists(target_dir):
            backup_dir = f"{os.path.abspath(target_dir)}.bak-{os.getpid()}"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.replace(target_dir, backup_dir)
        try:
            os.replace(staging_dir, target_dir)
        except BaseException:
            if backup_dir is not None and not os.path.exists(target_dir):
                os.replace(backup_dir, target_dir)
                backup_dir = None
            raise
        staging_dir = None
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)

    log.info(
        "Resharded `%s` -> `%s` (layout=%s, tensors=%d, size=%.2f MiB)",
        source_dir,
        target_dir,
        config.layout.value,
        len(refs),
        total_size / (1024 * 1024),
    )
    return target_dir
