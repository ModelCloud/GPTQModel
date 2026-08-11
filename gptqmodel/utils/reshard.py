# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Re-shard a safetensors checkpoint into a new per-layer or per-size layout.

Public API
----------
>>> from gptqmodel import reshard, ShardStrategy
>>> reshard(
...     "/path/to/Laguna-S-2.1",
...     "/path/to/Laguna-S-2.1-PER-LAYER",
...     strategy=ShardStrategy.PER_LAYER,
... )

Memory model
------------
The implementation is deliberately streaming: it parses all source shard
headers once to build a routing plan, then opens the source shards one at a
time.  Tensors are routed to per-output-group temporary partial files and the
source shard is closed before the next one is opened.  Final output shards are
merged from the partials in parallel using a bounded thread pool (up to
``num_write_workers`` shards in flight).  Peak host memory is therefore bounded by
a source shard (typically ~5 GB) plus ``num_write_workers`` output-group buffers,
not the size of the whole checkpoint.
"""

import json
import os
import shutil
import struct
import time
from collections import OrderedDict
from concurrent.futures import as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel.quantization.config import ShardStrategy
from gptqmodel.utils import log
from gptqmodel.utils.disk_telemetry import disk_telemetry
from gptqmodel.utils.model import get_checkpoints


_LAYER_CONTAINER_TOKENS = frozenset({"layers", "h", "blocks", "block", "encoder", "layer"})


def _resolve_layer_split_group(tensor_name: str, layer_prefixes: List[str]) -> tuple[str, bool]:
    """Return the layer group key and whether this tensor belongs to a layer."""
    for prefix in sorted((p for p in layer_prefixes if p), key=len, reverse=True):
        expected_prefix = f"{prefix}."
        if not tensor_name.startswith(expected_prefix):
            continue
        remainder = tensor_name[len(expected_prefix):]
        layer_idx, dot, _ = remainder.partition(".")
        if layer_idx.isdigit() and dot:
            return f"{prefix}.{layer_idx}", True

    if "." in tensor_name:
        return tensor_name.rsplit(".", 1)[0], False
    return "", False


def _layer_sort_key(group_name: str, layer_prefixes: List[str]) -> tuple:
    """Return a stable sort key for a layer group name like ``model.layers.0``."""
    for prefix in layer_prefixes:
        if group_name.startswith(f"{prefix}."):
            suffix = group_name[len(prefix) + 1:]
            try:
                return (layer_prefixes.index(prefix), int(suffix))
            except ValueError:
                pass
    return (len(layer_prefixes), group_name)


def _detect_layer_prefixes_from_names(weight_map: Dict[str, str]) -> List[str]:
    """Infer likely layer prefixes directly from tensor names.

    The first numeric segment in a tensor path is almost always the layer index
    for standard transformer checkpoints. We collect all such prefixes, prefer
    those whose final path component looks like a layer container, and return
    the distinct prefixes sorted longest-first.
    """
    prefix_stats: Dict[str, Dict[str, Any]] = {}
    for name in weight_map:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                prefix = ".".join(parts[:i])
                suffix = ".".join(parts[i + 1:]) if i + 1 < len(parts) else ""
                stats = prefix_stats.setdefault(prefix, {"count": 0, "suffixes": set()})
                stats["count"] += 1
                if suffix:
                    stats["suffixes"].add(suffix.split(".")[0])
                break

    if not prefix_stats:
        return []

    by_count = sorted(prefix_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)
    max_count = by_count[0][1]["count"]
    threshold = max(2, max_count // 10)

    candidates: List[str] = []
    for prefix, stats in prefix_stats.items():
        if stats["count"] < threshold:
            continue
        container = prefix.split(".")[-1].lower()
        looks_like_layer = (
            container in _LAYER_CONTAINER_TOKENS
            or len(stats["suffixes"]) > 1
        )
        if looks_like_layer:
            candidates.append(prefix)

    return sorted(candidates, key=len, reverse=True)


def _detect_layer_prefixes(
    source_path: str,
    weight_map: Dict[str, str],
    trust_remote_code: bool = False,
) -> List[str]:
    """Determine layer prefixes from the model definition when possible.

    Falls back to name-based heuristics when the model type is unsupported or
    the checkpoint has no usable ``config.json``.
    """
    try:
        from ..models.auto import check_and_get_model_definition

        model_cls = check_and_get_model_definition(
            source_path,
            trust_remote_code=trust_remote_code,
        )
        prefixes = model_cls.extract_layers_node() or []
        if prefixes:
            log.info(
                "Reshard: detected layer prefixes from model definition: %s",
                prefixes,
            )
            return prefixes
    except Exception as exc:
        log.debug(
            "Reshard: could not load layer prefixes from model definition: %s",
            exc,
        )

    prefixes = _detect_layer_prefixes_from_names(weight_map)
    if prefixes:
        log.info("Reshard: detected layer prefixes from tensor names: %s", prefixes)
    return prefixes


def _read_safetensors_header(shard_path: str) -> Dict[str, Any]:
    """Read only the header of a .safetensors file without loading tensors."""
    with open(shard_path, "rb") as fp:
        (header_len,) = struct.unpack("<Q", fp.read(8))
        raw = fp.read(header_len)
    return json.loads(raw.decode("utf-8"))


def _normalize_safetensors_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Convert arbitrary metadata values into the strings required by safetensors."""
    normalized: Dict[str, str] = {}
    for key, value in (metadata or {}).items():
        try:
            normalized[str(key)] = str(value)
        except Exception as exc:
            log.warn("Skipping safetensors metadata key %r that cannot be stringified: %s", key, exc)
    return normalized


def _load_weight_map(source_path: str) -> Tuple[Dict[str, str], bool]:
    """Load the source checkpoint index and return a {tensor: filename} map."""
    is_sharded, resolved_archive_file, _ = get_checkpoints(
        source_path,
        extensions=[".safetensors"],
        possible_model_basenames=["model", "pytorch_model"],
    )

    if is_sharded:
        with open(resolved_archive_file, encoding="utf-8") as fp:
            index = json.load(fp)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index in {source_path}")
        return {str(k): str(v) for k, v in weight_map.items()}, is_sharded

    # Single-file checkpoint: build a map pointing every key to the one shard.
    shard_name = os.path.basename(resolved_archive_file)
    with safe_open(resolved_archive_file, framework="pt", device="cpu") as handler:
        keys = list(handler.keys())
    return {str(k): shard_name for k in keys}, is_sharded


def _pack_subgroups(
    names: List[str],
    tensor_sizes: Dict[str, int],
    max_bytes: Optional[int],
) -> List[List[str]]:
    """Pack tensor names into output-shard subgroups respecting a byte cap."""
    if max_bytes is None or max_bytes <= 0:
        return [names]

    subgroups: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_sizes.get(name, 0)
        if current and current_bytes + size > max_bytes:
            subgroups.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        subgroups.append(current)
    return subgroups


def _pack_routed_module_subgroups(
    names: List[str],
    tensor_sizes: Dict[str, int],
    max_bytes: Optional[int],
    *,
    layer_group: str,
    routed_module_templates: Iterable[str],
) -> List[List[str]]:
    """Pack routed modules atomically so one module's state is never split."""

    if max_bytes is None or max_bytes <= 0:
        return [names]

    module_names: Dict[str, List[str]] = OrderedDict()
    for name in names:
        identity = _routed_module_identity(name, layer_group, routed_module_templates)
        if identity is None:
            raise RuntimeError(
                f"Routed shard group for {layer_group!r} contains an untagged tensor: {name!r}"
            )
        module_names.setdefault(identity, []).append(name)

    subgroups: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for state_names in module_names.values():
        module_bytes = sum(tensor_sizes.get(name, 0) for name in state_names)
        if current and current_bytes + module_bytes > max_bytes:
            subgroups.append(current)
            current = []
            current_bytes = 0
        current.extend(state_names)
        current_bytes += module_bytes
    if current:
        subgroups.append(current)
    return subgroups


def routed_module_templates_from_model_definition(model_cls: type) -> List[str]:
    """Return routed projection templates declared explicitly by ``module_tree``."""
    from ..models.base import MODULE_TREE_FLAG_ROUTED

    model_cls.build_layer_modules(model_cls.module_tree)
    metadata = model_cls._module_tree_metadata_cache.get(model_cls, {})
    return sorted(path for path, item in metadata.items() if MODULE_TREE_FLAG_ROUTED in item.flags)


def _match_module_template(relative_name: str, template: str) -> Optional[str]:
    """Return the concrete module identity when a state key matches a template."""
    name_parts = relative_name.split(".")
    template_parts = template.split(".")
    if len(name_parts) <= len(template_parts):
        return None
    for actual, expected in zip(name_parts, template_parts):
        if expected in ("#", "{expert_index}"):
            if not actual.isdigit():
                return None
        elif actual != expected:
            return None
    return ".".join(name_parts[: len(template_parts)])


def _routed_module_identity(
    tensor_name: str,
    layer_group: str,
    routed_module_templates: Iterable[str],
) -> Optional[str]:
    prefix = f"{layer_group}."
    if not tensor_name.startswith(prefix):
        return None
    relative_name = tensor_name[len(prefix):]
    for template in routed_module_templates:
        identity = _match_module_template(relative_name, template)
        if identity is not None:
            return identity
    return None


def _group_per_layer_names(
    names: Iterable[str],
    *,
    layer_prefixes: List[str],
    strategy: ShardStrategy,
    routed_module_templates: Optional[List[str]] = None,
    moe_modules_per_shard: int = 128,
) -> tuple[Dict[str, List[str]], Dict[str, bool]]:
    """Build deterministic per-layer groups without inferring MoE roles from names."""
    if moe_modules_per_shard < 1:
        raise ValueError("moe_modules_per_shard must be positive")
    if strategy is ShardStrategy.PER_LAYER_MOE and not routed_module_templates:
        raise ValueError(
            "PER_LAYER_MOE requires routed module templates from the model definition's module_tree"
        )

    layer_dense: Dict[str, List[str]] = OrderedDict()
    layer_routed: Dict[str, Dict[str, List[str]]] = OrderedDict()
    non_layer: List[str] = []
    for name in sorted(names):
        layer_group, is_layer = _resolve_layer_split_group(name, layer_prefixes)
        if not is_layer:
            non_layer.append(name)
            continue
        identity = None
        if strategy is ShardStrategy.PER_LAYER_MOE:
            identity = _routed_module_identity(name, layer_group, routed_module_templates or [])
        if identity is None:
            layer_dense.setdefault(layer_group, []).append(name)
        else:
            layer_routed.setdefault(layer_group, OrderedDict()).setdefault(identity, []).append(name)

    group_names: Dict[str, List[str]] = OrderedDict()
    group_is_layer: Dict[str, bool] = {}
    layer_keys = sorted(set(layer_dense) | set(layer_routed), key=lambda s: _layer_sort_key(s, layer_prefixes))
    for layer_group in layer_keys:
        dense_names = layer_dense.get(layer_group, [])
        if dense_names:
            group_names[layer_group] = dense_names
            group_is_layer[layer_group] = True
        routed = layer_routed.get(layer_group, {})
        identities = sorted(routed)
        for start in range(0, len(identities), moe_modules_per_shard):
            chunk = identities[start : start + moe_modules_per_shard]
            group = f"{layer_group}.routed.{start // moe_modules_per_shard:04d}"
            group_names[group] = [name for identity in chunk for name in routed[identity]]
            group_is_layer[group] = True
    if non_layer:
        group_names["non_layer"] = non_layer
        group_is_layer["non_layer"] = False
    return group_names, group_is_layer


def _copy_metadata_files(source_path: str, target_path: str) -> List[str]:
    """Copy non-safetensors config / tokenizer files to the target directory."""
    copied: List[str] = []
    for entry in os.listdir(source_path):
        src = os.path.join(source_path, entry)
        if not os.path.isfile(src):
            continue
        lower = entry.lower()
        if lower.endswith(".safetensors") or lower.endswith(".index.json"):
            continue
        dst = os.path.join(target_path, entry)
        shutil.copy2(src, dst)
        copied.append(entry)
    return copied


def _partial_filename(
    group: str,
    subgroup: int,
    source_idx: int,
) -> str:
    """Stable filename for a temporary partial shard."""
    return f"grp_{group}_sub{subgroup}_src{source_idx:05d}.safetensors"


def reshard(
    source_path: str,
    target_path: str,
    *,
    strategy: ShardStrategy = ShardStrategy.PER_LAYER,
    layer_prefixes: Optional[List[str]] = None,
    trust_remote_code: bool = False,
    max_shard_size_gb: Optional[float] = None,
    overwrite: bool = False,
    progress: bool = True,
    num_write_workers: int = 8,
    moe_modules_per_shard: int = 128,
) -> Dict[str, Any]:
    """Re-shard a safetensors checkpoint into a new directory.

    Args:
        source_path: directory containing the source safetensors checkpoint.
        target_path: directory to create with the new sharded checkpoint.
        strategy: per-layer sharding strategy. ``PER_LAYER_MOE`` separates
            dense/shared tensors from bounded routed-expert projection groups.
        layer_prefixes: optional layer node prefix(es) to match. Each prefix
            should be a dot-separated module path without a trailing dot (e.g.
            ``["model.layers"]`` or ``["language_model.model.layers"]``).
            If ``None``, prefixes are auto-detected from ``config.json`` and the
            matching ``GPTQModel`` definition, falling back to tensor-name
            heuristics.
        trust_remote_code: passed to ``AutoConfig.from_pretrained`` when
            loading ``config.json`` to determine the model definition.
        max_shard_size_gb: optional cap on each output shard size in GB. If a
            group is larger it is split across multiple sequentially-numbered
            shards.
        overwrite: if ``True`` and ``target_path`` exists, remove it first.
        progress: if ``True``, emit a LogBar progress bar and periodic telemetry.
        num_write_workers: number of parallel threads used when merging partial
            files and writing the final output shards. Default is ``8``; set to
            ``1`` to recover the previous serial write behavior.
        moe_modules_per_shard: maximum routed projection modules in each MoE
            shard for ``PER_LAYER_MOE``. All state tensors for a module remain
            together.

    Returns:
        A dictionary with ``output_files``, ``num_layers``, ``num_tensors``,
        ``total_bytes``, ``elapsed_seconds``, and ``throughput_mbps``.
    """
    start = time.perf_counter()

    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Source checkpoint directory not found: {source_path}")

    if not isinstance(strategy, ShardStrategy):
        try:
            strategy = ShardStrategy(strategy)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Unknown shard strategy: {strategy!r}") from exc

    src_real = os.path.realpath(source_path)
    tgt_real = os.path.realpath(target_path)
    if src_real == tgt_real or os.path.commonpath([src_real, tgt_real]) == tgt_real:
        raise ValueError(
            f"Target path {target_path!r} cannot be the same as or a parent of source path {source_path!r}."
        )

    if os.path.exists(target_path):
        if overwrite:
            shutil.rmtree(target_path)
        else:
            raise FileExistsError(
                f"Target directory already exists: {target_path}. "
                f"Pass overwrite=True to replace it."
            )
    os.makedirs(target_path, exist_ok=True)

    max_bytes = None
    if max_shard_size_gb is not None and max_shard_size_gb > 0:
        max_bytes = int(max_shard_size_gb * 1024**3)

    log.info("Resharding checkpoint from %s to %s", source_path, target_path)

    weight_map, is_sharded = _load_weight_map(source_path)
    if not weight_map:
        raise ValueError(f"No tensors found in source checkpoint: {source_path}")

    # Parse headers from every source shard to get tensor byte sizes without
    # touching the actual weight data.  This lets us build the output plan and
    # assign filenames before we stream any tensors.
    source_files = sorted(set(weight_map.values()))
    tensor_sizes: Dict[str, int] = {}
    source_metadata: Dict[str, Any] = {}
    for shard_file in source_files:
        shard_path = os.path.join(source_path, shard_file)
        header = _read_safetensors_header(shard_path)
        for name, info in header.items():
            if name == "__metadata__":
                if isinstance(info, dict):
                    source_metadata.update(info)
                continue
            if not isinstance(info, dict):
                continue
            offsets = info.get("data_offsets")
            if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
                tensor_sizes[name] = int(offsets[1]) - int(offsets[0])

    output_metadata = _normalize_safetensors_metadata(source_metadata)
    output_metadata["format"] = "pt"

    missing_size = set(weight_map.keys()) - set(tensor_sizes.keys())
    if missing_size:
        raise RuntimeError(
            f"Could not determine sizes for {len(missing_size)} tensor(s): "
            f"{sorted(missing_size)[:5]}"
        )

    # Determine layer prefixes from the model definition, or fall back to
    # name-based heuristics if no model definition is available.
    model_cls = None
    if strategy is ShardStrategy.PER_LAYER_MOE:
        try:
            from ..models.auto import check_and_get_model_definition

            model_cls = check_and_get_model_definition(source_path, trust_remote_code=trust_remote_code)
        except Exception as exc:
            raise ValueError(
                "PER_LAYER_MOE requires a supported model definition with explicit routed module-tree tags"
            ) from exc

    if layer_prefixes is None:
        layer_prefixes = _detect_layer_prefixes(
            source_path,
            weight_map,
            trust_remote_code=trust_remote_code,
        )
        # Final backward-compatible fallback if nothing was detected.
        if not layer_prefixes:
            layer_prefixes = ["model.layers"]
    elif isinstance(layer_prefixes, str):
        layer_prefixes = [layer_prefixes]
    else:
        layer_prefixes = list(layer_prefixes)

    if not layer_prefixes or any(not isinstance(p, str) or not p for p in layer_prefixes):
        raise ValueError("layer_prefixes must be a non-empty string or list of non-empty strings")

    routed_templates = (
        routed_module_templates_from_model_definition(model_cls)
        if model_cls is not None
        else None
    )
    group_names, group_is_layer = _group_per_layer_names(
        weight_map,
        layer_prefixes=layer_prefixes,
        strategy=strategy,
        routed_module_templates=routed_templates,
        moe_modules_per_shard=moe_modules_per_shard,
    )
    transformer_layer_keys = sorted(
        {
            group
            for name in weight_map
            for group, is_layer in [_resolve_layer_split_group(name, layer_prefixes)]
            if is_layer
        },
        key=lambda s: _layer_sort_key(s, layer_prefixes),
    )

    layer_keys = sorted(
        (k for k in group_names if group_is_layer.get(k)),
        key=lambda s: _layer_sort_key(s, layer_prefixes),
    )
    non_layer_keys = [k for k in group_names if not group_is_layer.get(k)]
    ordered_groups = layer_keys + non_layer_keys

    if not layer_keys:
        raise ValueError(
            f"No layer tensors matched layer_prefixes={layer_prefixes!r}; "
            "cannot produce per-layer shards."
        )

    log.info(
        "Reshard plan: %d transformer layer(s), %d layer-local shard group(s), %d non-layer tensor(s)",
        len(transformer_layer_keys),
        len(layer_keys),
        len(group_names.get("non_layer", [])),
    )

    # Build output subgroups and pre-assign filenames.  Each subgroup becomes one
    # final output shard; a group may be split if max_shard_size_gb is set.
    OutputSubgroup = Tuple[str, List[str], str]  # group, names, final_filename
    output_plan: List[OutputSubgroup] = []
    name_to_subgroup: Dict[str, Tuple[str, int]] = {}  # name -> (group, subgroup_index)
    subgroup_partial_files: Dict[Tuple[str, int], List[str]] = {}

    shard_counter = 1
    for group in ordered_groups:
        names = group_names[group]
        if strategy is ShardStrategy.PER_LAYER_MOE and ".routed." in group:
            layer_group = group.rsplit(".routed.", 1)[0]
            subgroups = _pack_routed_module_subgroups(
                names,
                tensor_sizes,
                max_bytes,
                layer_group=layer_group,
                routed_module_templates=routed_templates or (),
            )
        else:
            subgroups = _pack_subgroups(names, tensor_sizes, max_bytes)
        for subgroup_idx, subgroup_names in enumerate(subgroups):
            final_filename = f"model-{shard_counter:05d}-of-XXX.safetensors"
            output_plan.append((group, subgroup_names, final_filename))
            for name in subgroup_names:
                name_to_subgroup[name] = (group, subgroup_idx)
            subgroup_partial_files[(group, subgroup_idx)] = []
            shard_counter += 1

    total_output_shards = len(output_plan)
    # Replace placeholder total shard count in the filenames.
    final_plan: List[Tuple[str, List[str], str]] = []
    final_filename_by_subgroup: Dict[Tuple[str, int], str] = {}
    for group, names, placeholder in output_plan:
        # Extract the original counter from the placeholder.
        counter = int(placeholder.split("-")[1])
        final_filename = f"model-{counter:05d}-of-{total_output_shards:05d}.safetensors"
        final_plan.append((group, names, final_filename))
        subgroup_idx = name_to_subgroup[names[0]][1]
        final_filename_by_subgroup[(group, subgroup_idx)] = final_filename

    subgroup_sources = {
        subgroup_key: {weight_map[name] for name in names}
        for group, names, _ in final_plan
        for subgroup_key in [(group, name_to_subgroup[names[0]][1])]
    }
    direct_output_paths: Dict[Tuple[str, int], str] = {}

    # Staging area for partial files, kept under the target directory on the
    # same filesystem so rename is cheap and cleanup is simple.
    staging_dir = os.path.join(target_path, ".reshard_staging")
    os.makedirs(staging_dir, exist_ok=True)

    try:
        if progress:
            read_pb = log.pb(range(len(source_files))).manual().set(show_left_steps=False)
            read_pb.title(f"Reading source shards ({len(source_files)})")
            read_pb.subtitle("opening shard")
            read_pb.draw(force=True)
        else:
            read_pb = None

        # Phase 1: open source shards one at a time, route tensors to per-output
        # subgroup partial files, then close the source shard.
        phase1_start = time.perf_counter()
        for source_idx, shard_file in enumerate(source_files, start=1):
            shard_path = os.path.join(source_path, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Source shard missing: {shard_path}")

            # Determine which output subgroups need tensors from this shard.
            relevant: Dict[Tuple[str, int], List[str]] = {}
            for name, src_file in weight_map.items():
                if src_file != shard_file:
                    continue
                subgroup_key = name_to_subgroup[name]
                relevant.setdefault(subgroup_key, []).append(name)

            if not relevant:
                log.warning("No tensors from source shard %s appear in weight map", shard_file)
                if read_pb:
                    read_pb.next()
                    read_pb.subtitle(shard_file)
                    read_pb.draw()
                continue

            shard_start = time.perf_counter()
            with safe_open(shard_path, framework="pt", device="cpu") as handler:
                # Fill per-subgroup buffers from this source shard.
                subgroup_buffers: Dict[Tuple[str, int], Dict[str, Any]] = {
                    key: {} for key in relevant
                }
                for subgroup_key, names in relevant.items():
                    for name in names:
                        subgroup_buffers[subgroup_key][name] = handler.get_tensor(name)

            def _write_source_subgroup(subgroup_key: Tuple[str, int]) -> Tuple[Tuple[str, int], str, bool]:
                group, subgroup_idx = subgroup_key
                if len(subgroup_sources[subgroup_key]) == 1:
                    output_path = os.path.join(target_path, final_filename_by_subgroup[subgroup_key])
                    save_file(subgroup_buffers[subgroup_key], output_path, metadata=output_metadata)
                    return subgroup_key, output_path, True
                partial_name = _partial_filename(group, subgroup_idx, source_idx)
                partial_path = os.path.join(staging_dir, partial_name)
                save_file(subgroup_buffers[subgroup_key], partial_path, metadata=output_metadata)
                return subgroup_key, partial_path, False

            # Use the process-wide ThreadX loader lanes. Besides avoiding
            # temporary executors, independent output files can drain the same
            # source mmap concurrently instead of serializing dozens of MoE
            # groups behind one Python thread.
            from gptqmodel import DEVICE_THREAD_POOL

            futures = {
                DEVICE_THREAD_POOL.submit("model_loader:cpu", _write_source_subgroup, subgroup_key): subgroup_key
                for subgroup_key in relevant
            }
            for future in as_completed(futures):
                subgroup_key, output_path, is_direct = future.result()
                if is_direct:
                    direct_output_paths[subgroup_key] = output_path
                else:
                    subgroup_partial_files[subgroup_key].append(output_path)
                del subgroup_buffers[subgroup_key]

            shard_size = os.path.getsize(shard_path)
            shard_elapsed = time.perf_counter() - shard_start
            disk_telemetry.record_read(shard_size, shard_elapsed, source="reshard")
            log.info(
                "Source shard %d/%d: %s routed %d tensor(s) to %d subgroup(s) in %.2fs (%s/s)",
                source_idx,
                len(source_files),
                shard_file,
                sum(len(v) for v in relevant.values()),
                len(relevant),
                shard_elapsed,
                _fmt_bytes(shard_size / shard_elapsed) if shard_elapsed > 0 else "N/A",
            )

            if read_pb:
                read_pb.next()
                read_pb.subtitle(shard_file)
                read_pb.draw()

        log.info(
            "Phase 1 complete: %d source shard(s) routed in %.2fs",
            len(source_files),
            time.perf_counter() - phase1_start,
        )

        if progress:
            write_pb = log.pb(range(total_output_shards)).manual().set(show_left_steps=False)
            write_pb.title(f"Writing output shards ({total_output_shards})")
            write_pb.subtitle("merging partials")
            write_pb.draw(force=True)
        else:
            write_pb = None

        # Phase 2: merge partial files into final output shards in parallel.
        new_weight_map: Dict[str, str] = {}
        output_files: List[str] = []
        total_written = 0

        def _write_output_shard(args: Tuple[str, List[str], str]) -> Tuple[str, int, float, List[str]]:
            """Worker: merge partials for one output shard and write it."""
            group, subgroup_names, final_filename = args
            subgroup_idx = name_to_subgroup[subgroup_names[0]][1]
            subgroup_key = (group, subgroup_idx)
            direct_path = direct_output_paths.get(subgroup_key)
            if direct_path is not None:
                return final_filename, os.path.getsize(direct_path), 0.0, subgroup_names
            partial_paths = subgroup_partial_files[subgroup_key]
            merged: Dict[str, Any] = {}
            for partial_path in partial_paths:
                merged.update(load_file(partial_path))

            output_path = os.path.join(target_path, final_filename)
            write_start = time.perf_counter()
            save_file(merged, output_path, metadata=output_metadata)
            write_elapsed = time.perf_counter() - write_start
            shard_size = os.path.getsize(output_path)
            del merged
            return final_filename, shard_size, write_elapsed, subgroup_names

        # Bound outstanding work so ThreadX's shared loader lanes do not retain
        # every merged shard buffer at once.
        for plan_start in range(0, len(final_plan), max(1, num_write_workers)):
            plan_batch = final_plan[plan_start : plan_start + max(1, num_write_workers)]
            futures = {
                DEVICE_THREAD_POOL.submit("model_loader:cpu", _write_output_shard, item): item
                for item in plan_batch
            }
            for future in as_completed(futures):
                final_filename, shard_size, write_elapsed, subgroup_names = future.result()

                group = futures[future][0]
                disk_telemetry.record_write(shard_size, write_elapsed)
                total_written += shard_size

                for name in subgroup_names:
                    new_weight_map[name] = final_filename
                output_files.append(final_filename)

                log.info(
                    "Output shard %d/%d: %s (%s, %d tensor(s)) in %.2fs (%s/s)",
                    len(output_files),
                    total_output_shards,
                    final_filename,
                    _fmt_bytes(shard_size),
                    len(subgroup_names),
                    write_elapsed,
                    _fmt_bytes(shard_size / write_elapsed) if write_elapsed > 0 else "N/A",
                )

                if write_pb:
                    write_pb.next()
                    write_pb.subtitle(f"{final_filename} | {group}")
                    write_pb.draw()

    finally:
        # Always clean up staging, even on failure.
        shutil.rmtree(staging_dir, ignore_errors=True)

    # Copy config / tokenizer files and write the new index.
    metadata_copied = _copy_metadata_files(source_path, target_path)
    index_path = os.path.join(target_path, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "metadata": {"total_size": total_written},
                "weight_map": new_weight_map,
            },
            fp,
            indent=2,
            sort_keys=True,
        )

    elapsed = time.perf_counter() - start
    result = {
        "source_path": source_path,
        "target_path": target_path,
        "strategy": strategy.value,
        "is_sharded": is_sharded,
        "num_layers": len(transformer_layer_keys),
        "num_layer_shard_groups": len(layer_keys),
        "num_non_layer_tensors": len(group_names.get("non_layer", [])),
        "num_tensors": len(new_weight_map),
        "num_shards": total_output_shards,
        "moe_modules_per_shard": moe_modules_per_shard if strategy is ShardStrategy.PER_LAYER_MOE else None,
        "output_files": output_files,
        "copied_metadata": metadata_copied,
        "total_bytes": total_written,
        "elapsed_seconds": elapsed,
        "throughput_mbps": (total_written / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0,
    }

    log.info(
        "Reshard complete: %d shards, %s written in %.2fs (%.2f MB/s)",
        total_output_shards,
        _fmt_bytes(total_written),
        elapsed,
        result["throughput_mbps"],
    )

    disk_telemetry.log_summary(log, label="reshard", total_model_bytes=total_written, all_time=False)
    return result


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"
