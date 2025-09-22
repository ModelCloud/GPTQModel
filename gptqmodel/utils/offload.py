# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import os
import shutil
import threading
from typing import Iterable, List, Optional, Set, Tuple

import torch
# move base_module tensors to disk
from accelerate import disk_offload
from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules
from accelerate.utils import align_module_device, has_offloaded_params
from torch import nn

from ..looper.named_module import NamedModule
from .torch import CPU, META

_lock = threading.Lock()

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

def offload_to_disk(module: List[str] | nn.Module, model: nn.Module, disk_path: str = "." ):
    assert module is not None
    assert model is not None

    with _lock:
        if isinstance(module, List):
            for name in module:
                m = get_submodule(model, name)
                # unwrap named module
                if isinstance(m, NamedModule):
                    # print(f"offloading named module: {module.full_name}")
                    m = m.module

                full_name = get_module_fullname(model=model, module=m)
                _offload_disk(module=m, name=full_name, disk_path=disk_path)
        else:
            # unwrap named module
            if isinstance(module, NamedModule):
                # print(f"offloading named module: {module.full_name}")
                module = module.module

            full_name = get_module_fullname(model=model, module=module)
            
            if full_name != "":
                _offload_disk(module=module, name=full_name, disk_path=disk_path)

        if hasattr(module, "config") and hasattr(module.config,
                                                 "tie_word_embeddings") and module.config.tie_word_embeddings:
            module.tie_weights()  # makes lm_head.weight point to embed_tokens.weight again after offload

    # print("offload_disk: list item tree")
            # print_module_tree(module)

def _offload_disk(module: nn.Module, name: str, disk_path: str = "."):
    if is_meta_module(module):
        # print(f"[skip] '{name}' is on meta; leaving as-is")
        return

    # print(f"device_map base_modules: {device_map}")

    _ = disk_offload(
        module,
        # device_map={ "" : "disk" },  # only touch this subtree
        offload_dir=f"{disk_path}/{name}",
        offload_buffers=True,  # needed for buffers
        execution_device=CPU,
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


def _clone_into_parameter(t: torch.Tensor, *, device: torch.device, dtype: Optional[torch.dtype], requires_grad: bool) -> nn.Parameter:
    target = t
    if dtype is not None and target.dtype != dtype:
        target = target.to(dtype)
    if target.device != device:
        target = target.to(device, non_blocking=False)
    # clone to detach from any memory-mapped storage / hook-managed views
    target = target.detach().clone()
    return nn.Parameter(target, requires_grad=requires_grad)


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

    with torch.no_grad():
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
    with _lock:
        # Track candidate offload dirs if user asks to delete them later.
        offload_dirs: Set[str] = set()

        # 1) Materialize all offloaded leaves as real tensors on the target device/dtype.
        with torch.no_grad():
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
        remove_hook_from_submodules(module)      # public API
        remove_hook_from_module(module, recurse=False)  # ensure root is also clean

        # 3) Tie embedding if module is model and enabled/tied
        if hasattr(module, "config") and hasattr(module.config, "tie_word_embeddings") and module.config.tie_word_embeddings:
            module.tie_weights()  # makes lm_head.weight point to embed_tokens.weight again after undo_offload

        # 4) Optionally delete offload folders.
        if delete_offload_folders:
            for d in sorted(offload_dirs):
                with contextlib.suppress(Exception):
                    shutil.rmtree(d, ignore_errors=True)

        return module
