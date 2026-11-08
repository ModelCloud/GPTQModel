# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import hashlib
import json
import os
import shutil
import struct
from typing import Iterable, List, Optional, Set, Tuple

import accelerate
import torch

# move base_module tensors to disk
from accelerate import disk_offload
from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules
from accelerate.utils import align_module_device, has_offloaded_params
from safetensors.torch import save_file as safetensors_save_file
from torch import nn

from ..looper.named_module import NamedModule
from .device import get_device
from .module_locks import parent_module_lock
from .torch import CPU, META


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


_SAFETENSORS_DTYPE = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def offload_to_safetensors_reference(
    module: nn.Module,
    model: nn.Module,
    source_path: str | os.PathLike[str],
    disk_path: str = ".",
    module_name: str | None = None,
) -> dict:
    """Offload a module to an already-authenticated safetensors payload.

    The caller owns the source artifact and must validate its content before
    calling this helper. We validate the exact header/geometry, publish only a
    small Accelerate-compatible index, and replace the module tensors with META
    storage. The streaming model writer can then copy the original tensor ranges
    directly instead of materializing a second disk copy.
    """

    if module is None or model is None:
        raise ValueError("safetensors reference offload requires a module and model")
    source = os.path.abspath(os.fspath(source_path))
    if os.path.islink(source) or not os.path.isfile(source):
        raise ValueError("safetensors reference source is not a regular file")
    if module_name is None:
        full_name = get_module_fullname(model=model, module=module)
    elif not module_name or get_submodule(model, module_name) is not module:
        raise ValueError("safetensors reference module name differs from the model")
    else:
        full_name = module_name
    state = module.state_dict()
    if not state:
        raise ValueError("safetensors reference module has no persistent tensors")

    source_size = os.path.getsize(source)
    with open(source, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors reference source has no complete header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > source_size - 8:
            raise ValueError("safetensors reference source has an invalid header size")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("safetensors reference source has invalid JSON") from error
    if not isinstance(header, dict):
        raise ValueError("safetensors reference header is not an object")
    tensor_header = {
        key: value for key, value in header.items() if key != "__metadata__"
    }
    if set(tensor_header) != set(state):
        raise ValueError("safetensors reference tensor set differs from the module")

    data_base = 8 + header_size
    index: dict[str, dict] = {}
    ranges: list[tuple[int, int]] = []
    for key, tensor in state.items():
        entry = tensor_header.get(key)
        expected_dtype = _SAFETENSORS_DTYPE.get(tensor.dtype)
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or expected_dtype is None
            or entry.get("dtype") != expected_dtype
            or entry.get("shape") != list(tensor.shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise ValueError(
                f"safetensors reference metadata differs for tensor `{key}`"
            )
        start, end = offsets
        expected_bytes = tensor.numel() * tensor.element_size()
        if start < 0 or end - start != expected_bytes or data_base + end > source_size:
            raise ValueError(
                f"safetensors reference range differs for tensor `{key}`"
            )
        ranges.append((start, end))
        index[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "safetensors_file": source,
            "weight_name": key,
            "data_offsets": [data_base + start, data_base + end],
            "sha256": hashlib.sha256(
                memoryview(
                    tensor.detach()
                    .to(device="cpu")
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                )
            ).hexdigest(),
        }
    ordered_ranges = sorted(ranges)
    if (
        not ordered_ranges
        or ordered_ranges[0][0] != 0
        or any(
            left_end != right_start
            for (_, left_end), (right_start, _) in zip(
                ordered_ranges, ordered_ranges[1:]
            )
        )
        or data_base + ordered_ranges[-1][1] != source_size
    ):
        raise ValueError("safetensors reference tensor ranges are not contiguous")

    module_offload_dir = os.path.join(disk_path, full_name)
    _prepare_offload_directory(module_offload_dir)
    index_path = os.path.join(module_offload_dir, "index.json")
    temporary = f"{index_path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
        directory = os.open(module_offload_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

    module.to_empty(device=torch.device("meta"), recurse=True)
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
