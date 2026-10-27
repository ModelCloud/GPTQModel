# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import copy
import json
import os
import shutil
import struct
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

import accelerate
import torch

# move base_module tensors to disk
from accelerate import disk_offload
from accelerate.hooks import (
    AlignDevicesHook,
    SequentialHook,
    remove_hook_from_module,
    remove_hook_from_submodules,
)
from accelerate.utils import (
    PrefixedDataset,
    align_module_device,
    has_offloaded_params,
    offload_weight,
)
from safetensors.torch import save_file as safetensors_save_file
from torch import nn

from ..looper.named_module import NamedModule
from .device import get_device
from .logger import setup_logger
from .module_locks import parent_module_lock
from .torch import CPU, META


log = setup_logger()

_SMALL_MODULE_OFFLOAD_BYTES = 4 * 1024  # Skip disk writes for <4KB payloads


# Patch fix thread unsafe accelerate.utils.modeling.clear_device_cache
def _fake_clear_device_cache(garbage_collection=False):
    pass

# keep original
ACCELERATE_CLEAR_DEVICE_CACHE = accelerate.utils.modeling.clear_device_cache
accelerate.utils.modeling.clear_device_cache = _fake_clear_device_cache

def get_module_fullname(model: torch.nn.Module, module: torch.nn.Module) -> str:
    for name, mod in model.named_modules():
        if mod is module:
            return name  # dotted path like "model.embed_tokens" or "model.layers.0.self_attn.q_proj"

    name = module.full_name if module is NamedModule else ""
    raise Exception(f"module not found in model: name = {name}, module = {module}")

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
        if getattr(p, "is_meta", False) or (hasattr(p, "device") and p.device is META):
            return True
    for b in m.buffers(recurse=True):
        if hasattr(b, "device") and b.device is META:
            return True
    return False

# Serialize access to module.state_dict(), which is not thread-safe under
# concurrent calls that mutate the same parent module.
def _prepare_offload_directory(target_dir: str) -> None:
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        itemsize = tensor.element_size()
    except RuntimeError:
        itemsize = torch.empty((), dtype=tensor.dtype).element_size()
    return tensor.numel() * itemsize


def _bundle_module_state_dict(module: nn.Module, offload_dir: str, index_dir: Optional[str] = None) -> dict:
    """Serialize module state into offload_dir. index_dir is the directory
    the index.json should reference -- pass the final destination when writing
    into a temporary directory that will be atomically renamed into place."""
    bundle_path = os.path.join(offload_dir, "module.safetensors")
    index_bundle_path = os.path.join(index_dir, "module.safetensors") if index_dir else bundle_path
    index: dict[str, dict] = {}
    tensors: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        state_items = list(module.state_dict().items())

        for key, tensor in state_items:
            cpu_tensor = tensor.detach().to("cpu")
            tensors[key] = cpu_tensor.contiguous()
            entry = {
                "dtype": str(cpu_tensor.dtype).replace("torch.", ""),
                "shape": list(cpu_tensor.shape),
                "safetensors_file": os.path.abspath(index_bundle_path),
                "weight_name": key,
            }
            index[key] = entry

    safetensors_save_file(tensors, bundle_path)

    with open(bundle_path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        data_offset_base = fh.tell()

    for key, tensor_meta in header.items():
        if key == "__metadata__":
            continue
        entry = index.get(key)
        if entry is None:
            continue
        offsets = tensor_meta.get("data_offsets")
        if offsets is not None:
            start, end = (int(offsets[0]), int(offsets[1]))
            entry["data_offsets"] = [data_offset_base + start, data_offset_base + end]

    index_path = os.path.join(offload_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)

    return index


def offload_to_disk(module: List[str] | nn.Module, model: nn.Module, disk_path: str = ".", force: bool = False):
    """Offload module state to disk. force=True also writes modules below the
    small-module threshold -- required when the on-disk bundle is the only
    durable record (e.g. quantized modules that a resumed run must restore)."""
    _offload_to_disk_impl(module=module, model=model, disk_path=disk_path, force=force)


def _offload_to_disk_impl(module: List[str] | nn.Module, model: nn.Module, disk_path: str = ".", force: bool = False):
    assert module is not None
    assert model is not None

    #with _lock:
    if isinstance(module, List):
        for name in module:
            m = get_submodule(model, name)
            # unwrap named module
            if isinstance(m, NamedModule):
                # print(f"offloading named module: {module.full_name}")
                m = m.module

            full_name = get_module_fullname(model=model, module=m)
            _offload_disk(module=m, name=full_name, disk_path=disk_path, force=force)
    else:
        # unwrap named module
        if isinstance(module, NamedModule):
            # print(f"offloading named module: {module.full_name}")
            module = module.module

        full_name = get_module_fullname(model=model, module=module)

        _offload_disk(module=module, name=full_name, disk_path=disk_path, force=force)

    if hasattr(module, "config") and hasattr(module, "tie_weights") and getattr(module.config,
                                             "tie_word_embeddings", False):
        module.tie_weights()  # makes lm_head.weight point to embed_tokens.weight again after offload

    # print("offload_disk: list item tree")
            # print_module_tree(module)


# Serialize accelerate's disk hook mutations across threads.
#_OFFLOAD_SAFE = ThreadSafe(sys.modules[__name__])
#offload_to_disk = _OFFLOAD_SAFE.offload_to_disk

def _offload_disk(module: nn.Module, name: str, disk_path: str = ".", force: bool = False):
    with parent_module_lock(name):
        _offload_disk_locked(module=module, name=name, disk_path=disk_path, force=force)


def _offload_disk_locked(module: nn.Module, name: str, disk_path: str = ".", force: bool = False):
    # Finalizers are threads, protected by the parent-module lock above.
    # Checkpoint runs additionally hold a run lease and use private attempt
    # directories; unrelated modules do not need a global offload lock.
    if is_meta_module(module):
        # print(f"[skip] '{name}' is on meta; leaving as-is")
        return

    m_device = get_device(module)
    if m_device.type == "cuda":
        torch.cuda.set_device(m_device)

    # print(f"device_map base_modules: {device_map}")

    # skip modules that have no parameters and no buffers since they can't be offloaded
    has_params  = any(p.numel() > 0 for p in module.parameters(recurse=False))
    has_buffers = any(b.numel() > 0 for b in module.buffers(recurse=False))
    if not has_params and not has_buffers:
        return

    module_offload_dir = os.path.join(disk_path, name)

    total_bytes = 0

    state_items = list(module.state_dict().values())

    for tensor in state_items:
        total_bytes += _tensor_nbytes(tensor)

    if not force and total_bytes <= _SMALL_MODULE_OFFLOAD_BYTES:
        return

    # Write the new bundle into a sibling temp dir and swap it in with two
    # renames. The previous scheme (rmtree the live dir, then serialize into
    # it) had a wide crash window during which the only durable copy of the
    # module was destroyed -- fatal when that copy is a finalized quant
    # module a resumed run needs to restore. The index.json inside the
    # bundle must reference the final path, not the temp path (index_dir=).
    tmp_dir = f"{module_offload_dir}.tmp"
    old_dir = f"{module_offload_dir}.old"
    _prepare_offload_directory(tmp_dir)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)
    _bundle_module_state_dict(module, tmp_dir, index_dir=module_offload_dir)

    if os.path.isdir(module_offload_dir):
        os.rename(module_offload_dir, old_dir)
    os.rename(tmp_dir, module_offload_dir)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)

    _ = disk_offload(
        module,
        offload_dir=module_offload_dir,
        offload_buffers=True,
        execution_device=m_device,
    )

    # print("offload_disk: list item tree")
    # print_module_tree(module)

# undo offload
def _iter_leaf_tensors(mod: nn.Module, *, include_buffers: bool) -> Iterable[Tuple[str, torch.Tensor, bool]]:
    """Yield (name, tensor, is_param) for direct children (no recurse) to preserve module attribute names."""
    for n, p in mod.named_parameters(recurse=False):
        yield n, p, True
    if include_buffers:
        for n, b in mod.named_buffers(recurse=False):
            yield n, b, False


@contextlib.contextmanager
def _maybe_align(mod: nn.Module, device: torch.device):
    """
    If the module has offloaded params, temporarily align its tensors to `device`
    so we can clone them out as real tensors. If not offloaded, this is a no-op.
    """
    if has_offloaded_params(mod):  # public check
        with align_module_device(mod, execution_device=device):  # public context manager
            yield
    else:
        yield


@torch.inference_mode(False)
def _clone_into_parameter(t: torch.Tensor, *, device: torch.device, dtype: Optional[torch.dtype], requires_grad: bool) -> nn.Parameter:
    target = t
    if dtype is not None and target.dtype != dtype:
        target = target.to(dtype)
    if target.device != device:
        target = target.to(device, non_blocking=False)
    # clone to detach from any memory-mapped storage / hook-managed views
    target = target.detach().clone()
    return nn.Parameter(target, requires_grad=requires_grad)


@torch.inference_mode(False)
def _clone_into_buffer(t: torch.Tensor, *, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
    target = t
    if dtype is not None and target.dtype != dtype:
        target = target.to(dtype)
    if target.device != device:
        target = target.to(device, non_blocking=False)
    return target.detach().clone()


def _possible_offload_dirs_from_hook(mod: nn.Module) -> Set[str]:
    """
    Best-effort discovery of the on-disk folder used by Accelerate's offload weights_map.
    We *feature-detect* known attributes but never rely on them for correctness.
    """
    dirs: Set[str] = set()
    hook = getattr(mod, "_hf_hook", None)
    wm = getattr(hook, "weights_map", None) if hook is not None else None

    for attr in ("save_folder", "folder", "base_folder", "offload_dir"):
        val = getattr(wm, attr, None)
        if isinstance(val, (str, os.PathLike)) and os.path.isdir(val):
            dirs.add(os.fspath(val))

    ds = getattr(wm, "dataset", None)
    for attr in ("save_folder", "folder", "base_folder"):
        val = getattr(ds, attr, None)
        if isinstance(val, (str, os.PathLike)) and os.path.isdir(val):
            dirs.add(os.fspath(val))

    return dirs


def _restore_leaves_from_weights_map(mod: nn.Module, device: torch.device, dtype: Optional[torch.dtype]) -> bool:
    """
    Fast path: if this version of Accelerate exposes a per-module weights_map (as observed in
    multiple stacks), directly read tensors by name instead of going through a forward-time preloader.
    Returns True if handled, False to fall back to align+clone.
    """
    hook = getattr(mod, "_hf_hook", None)
    wm = getattr(hook, "weights_map", None) if hook is not None else None
    if wm is None:
        return False

    # Some implementations act like a Mapping[str, Tensor]; others expose .dataset.state_dict-like APIs.
    # We feature-detect Mapping behavior; otherwise bail out and let align+clone handle it.
    try:
        # Touch one known leaf name to see if subscript works; don't mutate anything yet.
        # Pick the first leaf (param or buffer) if available.
        sample = next(_iter_leaf_tensors(mod, include_buffers=True), None)
        if sample is None:
            return True  # nothing to restore for this module
        sample_name, _, _ = sample
        _ = wm[sample_name]  # may raise KeyError/TypeError if API is different
    except Exception:
        return False

    with torch.inference_mode():
        for name, tensor, is_param in list(_iter_leaf_tensors(mod, include_buffers=True)):
            is_meta = getattr(tensor, "is_meta", False) or tensor.device is META
            if not is_meta:
                continue  # already materialized
            try:
                src = wm[name]  # pull from offload map by fully-qualified leaf name
            except KeyError:
                # Not all buffers are necessarily offloaded; skip quietly.
                continue

            if is_param:
                new_p = _clone_into_parameter(src, device=device, dtype=dtype, requires_grad=tensor.requires_grad)
                setattr(mod, name, new_p)
            else:
                new_b = _clone_into_buffer(src, device=device, dtype=dtype)
                setattr(mod, name, new_b)

    return True


def _offload_hooks(module: nn.Module) -> Iterator[AlignDevicesHook]:
    """Yield this module's own ``AlignDevicesHook`` instances.

    ``has_offloaded_params`` only inspects a bare hook, so the ``SequentialHook``
    chains installed by ``from_pretrained`` are unwrapped here.
    """
    hook = getattr(module, "_hf_hook", None)
    pending = [hook] if hook is not None else []
    while pending:
        current = pending.pop(0)
        if isinstance(current, SequentialHook):
            pending = list(current.hooks) + pending
        elif isinstance(current, AlignDevicesHook):
            yield current


def _module_offload_hook(module: nn.Module) -> Optional[AlignDevicesHook]:
    """Return the offloading hook that owns this module's direct tensors, if any."""
    for hook in _offload_hooks(module):
        if hook.offload:
            return hook
    return None


class OffloadIndex:
    """Index of ancestor hooks that offload a nested module's tensors.

    ``dispatch_model`` normally attaches one hook per module that owns tensors, but
    ``preload_module_classes`` (full offload) attaches a single hook with
    ``place_submodules=True`` above them. Mapping every layer to that ancestor with
    a ``named_modules()`` walk per layer is quadratic for models with tens of
    thousands of quantized layers, so the mapping is built at most once and only
    when a module without its own hook asks for it.
    """

    def __init__(self, root: nn.Module) -> None:
        self._root = root
        self._targets: Optional[Dict[nn.Module, Tuple[nn.Module, AlignDevicesHook, str]]] = None

    def lookup(self, module: nn.Module) -> Optional[Tuple[nn.Module, AlignDevicesHook, str]]:
        """Return ``(hook_owner, hook, key_prefix)`` covering ``module``, if any."""
        if self._targets is None:
            self._build()
        return self._targets.get(module)

    def _build(self) -> None:
        root = self._root
        owners: List[Tuple[nn.Module, AlignDevicesHook, str]] = []
        for name, submodule in root.named_modules():
            for hook in _offload_hooks(submodule):
                if hook.offload and hook.place_submodules and hook.weights_map is not None:
                    owners.append((submodule, hook, name))

        targets: Dict[nn.Module, Tuple[nn.Module, AlignDevicesHook, str]] = {}
        # A deeper owner wins over an outer one, matching the hook chain that
        # accelerate installs for nested blocks. The root itself may own the hook
        # when the whole model is offloaded.
        for owner, hook, owner_name in sorted(owners, key=lambda item: item[2].count(".")):
            for name, submodule in owner.named_modules():
                prefix = "" if submodule is owner else f"{name}."
                targets[submodule] = (owner, hook, prefix)
        self._targets = targets


def offloaded_execution_device(
    module: nn.Module,
    root: Optional[nn.Module] = None,
    offload_index: Optional[OffloadIndex] = None,
) -> Optional[torch.device]:
    """Return the device an offloaded module's ``post_init()`` runs on, if any.

    Model-level setup such as ExllamaV2 scratch-space sizing reads the parameter
    device; for an offloaded layer that is ``meta``, so callers should ask here
    first and fall back to the real device only when no hook is present.
    """
    hook = _module_offload_hook(module)
    if hook is None and root is not None:
        index = offload_index if offload_index is not None else OffloadIndex(root)
        found = index.lookup(module)
        if found is not None:
            hook = found[1]
    if hook is None or hook.execution_device is None:
        return None
    return torch.device(hook.execution_device)


def _offload_map_targets(weights_map):
    """Split an accelerate ``weights_map`` into (prefix, state_dict, index, folder).

    ``PrefixedDataset`` (installed per module) wraps ``OffloadedWeightsLoader`` for
    disk offload or a plain dict for in-memory offload; ``cpu_offload``/plain
    ``dispatch_model`` state dicts are used as-is.
    """
    prefix = getattr(weights_map, "prefix", "")
    if not isinstance(prefix, str):
        prefix = ""
    dataset = getattr(weights_map, "dataset", weights_map)
    if isinstance(dataset, dict):
        return prefix, dataset, None, None
    state_dict = getattr(dataset, "state_dict", None)
    index = getattr(dataset, "index", None)
    folder = getattr(dataset, "save_folder", None)
    return (
        prefix,
        state_dict if isinstance(state_dict, dict) else None,
        index if isinstance(index, dict) else None,
        os.fspath(folder) if isinstance(folder, (str, os.PathLike)) else None,
    )


def _persist_offload_index(folder: str, entries: Mapping[str, dict]) -> None:
    """Best-effort update of ``folder/index.json`` so a refresh survives a reload."""
    index_path = os.path.join(folder, "index.json")
    try:
        current = {}
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fp:
                current = json.load(fp)
        current.update(entries)
        with open(index_path, "w", encoding="utf-8") as fp:
            json.dump(current, fp, indent=2)
    except OSError:  # pragma: no cover - read-only offload folder
        log.warn.once(
            "Offload: could not persist refreshed weight metadata; "
            "offloaded layers will need a reload of the checkpoint."
        )


def _refresh_weights_map(
    module: nn.Module,
    hook: AlignDevicesHook,
    snapshot: Mapping[str, torch.Tensor],
    key_prefix: str = "",
) -> None:
    """Replace stale offloaded entries with the post-``post_init`` tensors.

    ``accelerate.dispatch_model`` snapshots the weights *before* the quantizer runs,
    so kernels that rewrite their packed buffers in ``post_init()`` (Marlin clears
    and repacks ``g_idx``/``qzeros``/``qweight``/``scales``) leave an offload map
    that no longer matches the module's parameter shapes.  A stale map makes the
    next ``pre_forward`` call fail the shape check, or silently feed old weights.

    Entries are replaced with the packed quantized tensor, so no expanded
    (dequantized) weight copy is retained.
    """
    weights_map = hook.weights_map
    if weights_map is None or not snapshot:
        return

    prefix, state_dict, index, folder = _offload_map_targets(weights_map)
    if state_dict is None and index is None:
        raise RuntimeError(
            f"Cannot refresh offloaded weights for {type(module).__name__}: "
            f"accelerate's weights map ({type(weights_map).__name__}) is not writable."
        )

    persisted: Dict[str, dict] = {}
    for name, tensor in snapshot.items():
        key = f"{prefix}{key_prefix}{name}"
        # ``snapshot`` already owns a CPU clone of the packed tensor (see
        # ``_materialized_snapshot``), so reusing it here keeps peak memory at one
        # copy per layer instead of two.
        packed = tensor.detach().to("cpu").contiguous()

        if state_dict is not None and key in state_dict:
            # In-memory offload: the loader reads ``state_dict`` first, so this
            # supersedes any stale ``index`` entry without touching checkpoints.
            state_dict[key] = packed
            continue

        if index is not None and key in index:
            if folder is None:
                # Transformers can point an index entry straight at a checkpoint
                # file and keep no writable side table. Never write into the
                # checkpoint; ``OffloadedWeightsLoader`` reads ``state_dict``
                # first, so an empty one would silently pin the whole layer in
                # memory instead of on disk.
                if not state_dict:
                    raise RuntimeError(
                        f"Cannot refresh offloaded weights for {type(module).__name__}: "
                        f"accelerate's weights map has a read-only checkpoint entry "
                        f"for `{name}` and no offload folder to rewrite it in."
                    )
                state_dict[key] = packed
                continue
            os.makedirs(folder, exist_ok=True)
            # Rewriting the entry replaces the memory-mapped/checkpoint-backed
            # stale reference with packed storage in the offload folder.
            # ``offload_weight`` updates ``index`` in place and returns it, so
            # the entry must be read back instead of assigned from the result.
            offload_weight(packed, key, folder, index)
            persisted[key] = index[key]
            continue

        raise RuntimeError(
            f"Cannot refresh offloaded weights for {type(module).__name__}: "
            f"accelerate's weights map has no entry for `{name}`. The checkpoint "
            "must be reloaded with matching quantization metadata."
        )

    if persisted and folder is not None:
        _persist_offload_index(folder, persisted)


def _iter_persistent_tensors(module: nn.Module) -> Iterator[str]:
    """Yield names of persistent tensors owned directly by ``module``.

    Non-persistent runtime buffers are excluded: ``post_init()`` rebuilds them and
    accelerate keeps them (rather than the offload map) as their source of truth.
    """
    for name, _ in module.named_parameters(recurse=False):
        yield name
    for name, _ in module.named_buffers(recurse=False):
        if name not in module._non_persistent_buffers_set:
            yield name


@torch.inference_mode()
def _materialized_snapshot(module: nn.Module, names: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Copy the listed persistent tensors to CPU in their packed representation.

    The copy holds packed quantized storage (plus scales/indices), so refreshing an
    offload entry never installs an expanded (dequantized) weight cache.
    """
    snapshot: Dict[str, torch.Tensor] = {}
    missing: List[str] = []
    for name in names:
        tensor = getattr(module, name, None)
        if tensor is None:
            raise RuntimeError(
                f"{type(module).__name__}.post_init() removed the loaded tensor `{name}`; "
                "accelerate's offload map cannot be refreshed."
            )
        if tensor.is_meta:
            missing.append(name)
            continue
        snapshot[name] = tensor.detach().to("cpu").clone()
    if missing:
        raise RuntimeError(
            f"{type(module).__name__}.post_init() ran without materializing "
            f"offloaded tensor(s) {missing}; accelerate's weights map could not "
            "restore them, so the offloaded checkpoint must be reloaded."
        )
    return snapshot


@contextlib.contextmanager
def _materialize_offloaded_module(
    module: nn.Module, hook: AlignDevicesHook, key_prefix: str = ""
) -> Iterator[AlignDevicesHook]:
    """Temporarily load this module's offloaded tensors, then restore meta state.

    ``hook`` may cover an ancestor of ``module`` (``place_submodules=True``). It
    would then load every descendant, making one ``post_init()`` per layer
    quadratic in model size, so its weights map is rescaled onto ``module`` and
    only that layer is materialized.

    Yields the hook that actually performed the materialization.
    """
    active = hook
    if hook.place_submodules:
        weights_map = hook.weights_map
        prefix = getattr(weights_map, "prefix", None)
        dataset = getattr(weights_map, "dataset", None)
        if isinstance(prefix, str) and dataset is not None:
            active = copy.copy(hook)
            active.weights_map = PrefixedDataset(dataset, f"{prefix}{key_prefix}")
            active.place_submodules = False
    active.pre_forward(module)
    try:
        yield active
    finally:
        active.post_forward(module, None)


def post_init_offloaded_layer(
    module: nn.Module,
    root: Optional[nn.Module] = None,
    offload_index: Optional[OffloadIndex] = None,
    **post_init_kwargs,
) -> bool:
    """Run ``module.post_init()`` correctly when accelerate offloads its weights.

    ``from_pretrained`` dispatches the model with accelerate *before* the quantizer
    post-initializes the kernels.  For offloaded (``cpu``/``disk`` in ``device_map``)
    layers the packed tensors are on the meta device and the pre-init snapshot is
    captured in ``hook.weights_map``.  Kernels such as Marlin resize ``g_idx``,
    drop ``qzeros`` and repack ``qweight``/``scales``; that snapshot must be
    refreshed or the first forward fails the shape check (or reuses stale weights).

    ``root`` is the container whose accelerate hooks should be consulted when the
    offload hook lives on an ancestor. Pass a shared :class:`OffloadIndex` when
    calling this for every layer of one model to keep the lookup linear.

    Returns ``True`` when an offload hook handled the call, ``False`` when the
    caller should invoke ``post_init()`` directly.
    """
    owner, hook, key_prefix = module, _module_offload_hook(module), ""
    if hook is None and root is not None:
        index = offload_index if offload_index is not None else OffloadIndex(root)
        found = index.lookup(module)
        if found is not None:
            owner, hook, key_prefix = found

    if hook is None:
        return False

    offloaded = [
        name
        for name in _iter_persistent_tensors(module)
        if getattr(module, name).is_meta
    ]
    if not offloaded:
        # Loaded directly on its execution device: accelerate has no snapshot to
        # refresh, so the plain call is both correct and cheaper.
        return False

    snapshot: Dict[str, torch.Tensor] = {}
    with _materialize_offloaded_module(module, hook, key_prefix=key_prefix) as active:
        module.post_init(**post_init_kwargs)
        snapshot.update(_materialized_snapshot(module, offloaded))
    # The context restored meta parameters, so the refreshed packed tensors must
    # already be durable where the next ``pre_forward`` reads them from.
    # ``active.weights_map`` already carries ``key_prefix`` when the hook covered
    # an ancestor, so only the map's own prefix may be applied here.
    _refresh_weights_map(module, active, snapshot)
    return True


def run_post_init(
    module: nn.Module,
    root: Optional[nn.Module] = None,
    offload_index: Optional[OffloadIndex] = None,
    **post_init_kwargs,
) -> bool:
    """``module.post_init(**post_init_kwargs)`` that also handles offloaded layers.

    ``root`` is the container whose accelerate hooks should be consulted when the
    offload hook lives on an ancestor (``place_submodules=True``).

    Returns ``True`` when an offload hook handled the call, ``False`` when
    :func:`post_init_offloaded_layer` declined and ``post_init()`` ran directly.
    """
    if post_init_offloaded_layer(
        module, root=root, offload_index=offload_index, **post_init_kwargs
    ):
        return True
    module.post_init(**post_init_kwargs)
    return False


def undo_offload_to_disk(
    module: nn.Module,
    device: torch.device = CPU,
    include_buffers: bool = True,
    delete_offload_folders: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Reverse the effects of `accelerate.disk_offload` (or partial per-submodule disk offload) on `module`.

    What it does:
      1) Materializes all offloaded parameters (and optionally buffers) back into regular tensors on `device`.
      2) Detaches all Accelerate hooks on `module` and its submodules (restoring original `forward`).
      3) Optionally deletes discovered offload folders on disk.

    Args:
        module: Root module (can be the whole model or any sub-tree you disk_offloaded).
        device: Target device for restored tensors (default: CPU if unknown).
        dtype:  Optional dtype conversion for restored tensors (default: keep each tensor's original dtype).
        include_buffers: Restore buffers too (e.g., rotary caches) if they were offloaded.
        delete_offload_folders: Best-effort cleanup of the on-disk folders backing offloaded weights.

    Returns:
        The same `module`, now “de-offloaded”.
    """
    #with _lock:
    # Track candidate offload dirs if user asks to delete them later.
    offload_dirs: Set[str] = set()
    # Accelerate's detach_hook reloads offloaded leaves from weights_map onto
    # their original device and dtype, even after we materialize them below.
    # Remember meta leaves so we can honor the caller's requested target after
    # detaching without moving unrelated, already materialized model state.
    offloaded_leaves = [
        (sub, name, is_param, tensor.requires_grad)
        for sub in module.modules()
        for name, tensor, is_param in _iter_leaf_tensors(sub, include_buffers=include_buffers)
        if tensor.is_meta
    ]

    # 1) Materialize all offloaded leaves as real tensors on the target device/dtype.
    with torch.inference_mode():
        for sub in module.modules():
            if not has_offloaded_params(sub):
                continue

            # Discover offload folders opportunistically (optional cleanup)
            offload_dirs |= _possible_offload_dirs_from_hook(sub)

            # Prefer a fast path reading directly from the weights_map if exposed by this Accelerate version.
            handled = _restore_leaves_from_weights_map(sub, device=device, dtype=dtype)
            if handled:
                continue

            # Fallback path: ask Accelerate to align this submodule to the execution device,
            # then clone+rebind leaves so they become regular, hook-free tensors.
            with _maybe_align(sub, device=device):
                for name, tensor, is_param in list(_iter_leaf_tensors(sub, include_buffers=include_buffers)):
                    is_meta = (getattr(tensor, "is_meta", False) or tensor.device is META)
                    if not is_meta:
                        # Still clone if the hook attached a tensor view that would be re-offloaded later.
                        # Safer to always break links to hook-managed storages.
                        src = tensor
                    else:
                        # After align, meta leaves should be backed by real memory on `device`.
                        src = tensor

                    if is_param:
                        new_p = _clone_into_parameter(src, device=device, dtype=dtype, requires_grad=tensor.requires_grad)
                        setattr(sub, name, new_p)
                    else:
                        new_b = _clone_into_buffer(src, device=device, dtype=dtype)
                        setattr(sub, name, new_b)

        # 2) Remove all Accelerate hooks so future forwards won't offload again.
        # Detaching hooks can recreate tensors from their offload maps. Tying
        # weights may also update leaf parameters in place.
        with torch.inference_mode(False), torch.no_grad():
            remove_hook_from_submodules(module)      # public API
            remove_hook_from_module(module, recurse=False)  # ensure root is also clean

            for sub, name, is_param, requires_grad in offloaded_leaves:
                tensor = getattr(sub, name)
                if tensor.is_meta:
                    continue  # unrelated meta leaves have no offload hook to restore them
                if tensor.device == device and (dtype is None or tensor.dtype == dtype) and not tensor.is_inference():
                    continue
                if is_param:
                    restored = _clone_into_parameter(tensor, device=device, dtype=dtype, requires_grad=requires_grad)
                else:
                    restored = _clone_into_buffer(tensor, device=device, dtype=dtype)
                setattr(sub, name, restored)

            # 3) Tie embedding if module is model and enabled/tied
            if hasattr(module, "config") and getattr(module.config, "tie_word_embeddings", False):
                module.tie_weights()  # makes lm_head.weight point to embed_tokens.weight again after undo_offload

        # 4) Optionally delete offload folders.
        if delete_offload_folders:
            for d in sorted(offload_dirs):
                with contextlib.suppress(Exception):
                    shutil.rmtree(d, ignore_errors=True)

        return module
