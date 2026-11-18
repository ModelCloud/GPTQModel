# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import copy
import hashlib
import json
import os
import shutil
import struct
import uuid
import weakref
from collections.abc import Mapping
from typing import Iterable, List, Optional, Set, Tuple

import accelerate
import torch

# move base_module tensors to disk
from accelerate import disk_offload
from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules
from accelerate.utils import align_module_device, has_offloaded_params
from torch import nn

from ..looper.named_module import NamedModule
from .checkpoint_paths import resolve_checkpoint_path
from .device import get_device
from .logger import setup_logger
from .module_locks import parent_module_lock
from .safe import safetensors_save_file
from .tensor import tensor_buffer
from .tied_weights import retie_tied_weights, tied_weights_keys, word_embeddings_tied
from .torch import CPU, META


_SMALL_MODULE_OFFLOAD_BYTES = 4 * 1024  # Skip disk writes for <4KB payloads
# Bound on host bytes held while proving a live tensor equals its source payload.
_SOURCE_COMPARE_CHUNK_BYTES = 16 * 1024 * 1024

log = setup_logger()


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
        if p.is_meta:
            return True
    for b in m.buffers(recurse=True):
        if b.is_meta:
            return True
    return False


def bind_private_offload_root(model: nn.Module) -> Optional[str]:
    """Bind a private offload namespace to one quantization run.

    A ``QuantizeConfig`` (and therefore its automatic temporary directory or
    explicit ``offload_to_disk_path``) is commonly shared by several models:
    ``copy.copy``/``copy.deepcopy`` deliberately keep the same temporary
    directory. Bundle directory names are derived from the module path alone,
    so without a per-run namespace a later run overwrites the earlier model's
    finalized bundles in place and its ``save()`` silently writes the other
    model's weights.

    Give the run its own subdirectory under the configured root and keep a
    private config copy on the model, so a later run on another model cannot
    move this model's save-time bundle lookup. The directory is registered on
    the model so :func:`release_private_offload_root` can remove it after save,
    after a failed run, or when the model is garbage collected (issue #4368).
    """
    config = getattr(model, "quantize_config", None)
    if config is None or not getattr(config, "offload_to_disk", False):
        return None

    base = getattr(model, "_private_offload_base_root", None)
    if not base:
        configured = getattr(config, "offload_to_disk_path", None)
        if not configured:
            return None
        base = os.path.abspath(os.path.expanduser(os.fspath(configured)))

    # Re-quantizing the same model replaces its run state; release the previous
    # run directory instead of overwriting the only handle to it.
    if getattr(model, "_private_offload_run_root", None):
        release_private_offload_root(model)

    run_root = os.path.join(base, f"run-{uuid.uuid4().hex[:16]}")
    os.makedirs(run_root, exist_ok=True)

    private_config = copy.copy(config)
    private_config.offload_to_disk_path = run_root
    model.quantize_config = private_config
    model._private_offload_base_root = base
    model._private_offload_run_root = run_root
    try:
        model._private_offload_run_finalizer = weakref.finalize(
            model, _remove_private_offload_root, run_root
        )
    except TypeError:
        # Lightweight test doubles (for example ``SimpleNamespace``) are not
        # weak-referenceable; they can still call the explicit release helper.
        model._private_offload_run_finalizer = None
    log.debug("Offload: bound private run directory `%s`.", run_root)
    return run_root


def _remove_private_offload_root(run_root: str) -> None:
    """Remove one per-run offload directory without touching its siblings."""

    try:
        shutil.rmtree(run_root, ignore_errors=True)
    except Exception:  # pragma: no cover - defensive
        log.debug("Offload: failed to remove run directory `%s`.", run_root, exc_info=True)


def release_private_offload_root(model: nn.Module) -> Optional[str]:
    """Remove the private offload directory bound to ``model``, if any.

    The directory is only deleted after the caller has finished reading its
    bundles (save) or after the run failed. Keeping the explicit path's parent
    intact means other runs and the caller's own files are untouched.
    """

    finalizer = getattr(model, "_private_offload_run_finalizer", None)
    if finalizer is not None:
        try:
            finalizer.detach()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            model._private_offload_run_finalizer = None
        except Exception:  # pragma: no cover - defensive
            pass

    run_root = getattr(model, "_private_offload_run_root", None)
    if not run_root:
        return None
    try:
        model._private_offload_run_root = None
    except Exception:  # pragma: no cover - defensive
        pass
    _remove_private_offload_root(run_root)
    log.debug("Offload: released private run directory `%s`.", run_root)
    return run_root


def capture_offloaded_base_modules(
    model: nn.Module,
    module_names: Iterable[str],
) -> List[Tuple[str, nn.Module, torch.device]]:
    """Snapshot base modules before they are offloaded.

    Only modules that still hold real tensors are recorded: a module that is
    already on meta was not offloaded by this run and has no bundle to restore
    from.
    """
    entries: List[Tuple[str, nn.Module, torch.device]] = []
    for name in module_names:
        try:
            module = get_submodule(model, name)
        except AttributeError:
            continue
        if isinstance(module, NamedModule):
            module = module.module
        device = get_device(module)
        if device.type == "meta":
            continue
        entries.append((name, module, device))
    return entries


def restore_offloaded_modules(
    model: nn.Module,
    entries: Iterable[Tuple[str, nn.Module, torch.device]],
    *,
    strict: bool = False,
) -> List[str]:
    """Materialize modules offloaded during quantization back into memory.

    Quantization offloads the embedding/lm_head base modules so the layer loop
    has more host memory. They are never needed on disk afterwards, and leaving
    them on meta makes the in-session model unusable: ``tie_weights()`` points
    ``lm_head.weight`` at the meta embedding parameter, which has no offload
    hook. Restore the recorded modules and re-tie shared weights.

    ``strict=True`` raises a clear reload-the-checkpoint error when a module
    cannot be restored; otherwise the failure is only logged.
    """
    unresolved: List[str] = []
    for name, module, device in entries:
        # A base module can be replaced during quantization (the quantizer
        # swaps the recorded dense module for its packed kernel and writes the
        # packed bundle into the same offload directory). Restoring the stale
        # dense module would read the replacement's index and tensors, so skip
        # it: the live module owns its own bundle and is restored below by
        # `restore_disk_offloaded_modules` (issue #4758).
        try:
            live_module = get_submodule(model, name) if name else model
        except AttributeError:
            live_module = None
        if live_module is not module:
            log.debug(
                "Offload: `%s` was replaced during quantization; skipping the "
                "stale base-module restore.",
                name,
            )
            continue
        try:
            undo_offload_to_disk(module, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            log.warn("Offload: failed to restore `%s`: %s", name, exc)
            unresolved.append(name)
            continue
        if is_meta_module(module):
            unresolved.append(name)

    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        try:
            # Only re-apply the pairs the checkpoint really ties: a composite
            # wrapper can default `tie_word_embeddings` to True while the
            # decoder declares False and the checkpoint carries a real head
            # (#4799).
            retie_tied_weights(model)
        except Exception as exc:  # pragma: no cover - defensive
            log.warn("Offload: tie_weights() after restore failed: %s", exc)
            unresolved.append("<tie_weights>")

    if not unresolved:
        return []

    message = (
        "Offload: base modules remain on the meta device after quantization "
        f"({', '.join(unresolved)}). The in-session model cannot run inference; "
        "reload the saved checkpoint before using it."
    )
    if strict:
        raise RuntimeError(message)
    log.error(message)
    return unresolved

def _owns_offload_hook(module: nn.Module) -> bool:
    """Whether ``module`` itself carries an accelerate disk-offload hook."""

    pending = [getattr(module, "_hf_hook", None)]
    while pending:
        hook = pending.pop()
        if hook is None:
            continue
        nested = getattr(hook, "hooks", None)
        if isinstance(nested, (tuple, list)):
            pending.extend(nested)
        elif getattr(hook, "offload", False):
            return True
    return False


def has_disk_offloaded_modules(model: nn.Module) -> bool:
    """Whether any module still reads its weights from a disk-offload bundle.

    Such a module owns an accelerate disk-offload hook and points at a
    ``module.safetensors`` inside the run's private offload directory. That
    directory must outlive every module in this state: deleting it leaves the
    module on meta with a dangling hook, so the next forward raises
    ``FileNotFoundError`` (issue #4807).
    """

    return any(_owns_offload_hook(module) for module in model.modules())


def restore_disk_offloaded_modules(
    model: nn.Module,
    *,
    device: torch.device = CPU,
    strict: bool = False,
) -> List[str]:
    """Materialize every module still served by a disk-offload hook.

    During quantization each finalized (packed) module is written to the run's
    offload directory and replaced by meta tensors behind an accelerate hook,
    which keeps peak host memory at one layer. Left in place after the loop,
    those hooks re-read the bundles on every forward, make ``model.to()`` fail
    on the meta tensors, and break once the run directory is removed at
    teardown (issue #4450). Restore the modules into ordinary tensors on
    ``device`` and drop the hooks, so the in-session model is the same model
    ``offload_to_disk=False`` produces. The bundle's weights map hands GPTQ
    modules their live qzero layout (issue #4380).

    ``strict=True`` raises the reload-the-checkpoint error when a module keeps
    meta tensors; otherwise the failure is only logged. Returns the names of
    modules that could not be restored.
    """

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _owns_offload_hook(module)
    ]
    unresolved: List[str] = []
    for name, module in targets:
        try:
            undo_offload_to_disk(module, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            log.warn("Offload: failed to restore `%s`: %s", name, exc)
            unresolved.append(name)
            continue
        if is_meta_module(module) or _owns_offload_hook(module):
            unresolved.append(name)

    if not unresolved:
        return []

    message = (
        "Offload: quantized modules remain disk-offloaded after quantization "
        f"({', '.join(unresolved[:8])}{', ...' if len(unresolved) > 8 else ''}). "
        "The in-session model cannot run inference; reload the saved checkpoint "
        "before using it."
    )
    if strict:
        raise RuntimeError(message)
    log.error(message)
    return unresolved


def _owns_meta_tensor(module: nn.Module) -> bool:
    tensors = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
    return any(tensor.is_meta for tensor in tensors)


def _tracks_version(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` exposes a usable version counter.

    Inference tensors raise ``RuntimeError: Inference tensors do not track
    version counter``. A parameter can also lose its counter while reporting
    ``is_inference() == False``: ``post_quantize`` moves an inference-mode
    layer to CPU with ``param.data = cpu_copy``, and the resulting Parameter is
    an ordinary tensor that still cannot take part in in-place updates, so
    ``F.linear`` rejects it (issue #4572).
    """

    try:
        tensor._version  # noqa: SLF001 - the counter itself is the contract
    except RuntimeError:
        return False
    return True


@torch.inference_mode(False)
def replace_inference_parameters(module: nn.Module) -> None:
    """Rebind ``module``'s inference-mode parameters as ordinary parameters.

    The lazy checkpoint source materializes under ``torch.inference_mode``.
    ``nn.Module.to()`` updates parameters in place, which an inference tensor
    refuses outside that mode ("Inference tensors do not track version
    counter"), so a later ``model.to(device)`` on the in-session model failed.
    Buffers are reassigned by ``to()`` and need no rebinding.

    A parameter that lost its counter through ``.data =`` assignment is
    repaired too: it reports ``is_inference() == False`` yet still rejects the
    in-place update ``F.linear`` needs (issue #4572).
    """

    replacement: dict[int, nn.Parameter] = {}
    for submodule in module.modules():
        for name, param in list(submodule.named_parameters(recurse=False)):
            if not param.is_inference() and _tracks_version(param):
                continue
            fresh = replacement.get(id(param))
            if fresh is None:
                fresh = nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad)
                replacement[id(param)] = fresh
            setattr(submodule, name, fresh)


@torch.inference_mode(False)
def ensure_version_counters(model: nn.Module) -> List[str]:
    """Rebind every parameter whose version counter is unusable.

    This is the loop-exit repair for the #4498 contract: when ``quantize()``
    returns, every tensor in the in-session model must accept the in-place
    updates a forward performs. Quantized modules are restored into ordinary
    parameters already; unquantized layers (MoE routers, dynamic ``-:``
    exclusions, norms) were materialized under ``inference_mode`` and then
    moved to CPU with ``param.data = ...``, which leaves them with
    ``is_inference() == False`` and no counter (issue #4572).

    Aliased parameters keep their aliasing: every owner of one old object is
    rebound to the same replacement. Returns the rebind names for diagnostics.
    """

    replacement: dict[int, nn.Parameter] = {}
    repaired: List[str] = []
    for name, submodule in model.named_modules():
        for attr, param in list(submodule.named_parameters(recurse=False)):
            if param.is_inference() or not _tracks_version(param):
                fresh = replacement.get(id(param))
                if fresh is None:
                    fresh = nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad)
                    replacement[id(param)] = fresh
                setattr(submodule, attr, fresh)
                repaired.append(f"{name}.{attr}" if name else attr)
        for attr, buffer in list(submodule.named_buffers(recurse=False)):
            if buffer.is_inference() or not _tracks_version(buffer):
                setattr(submodule, attr, buffer.detach().clone())
                repaired.append(f"{name}.{attr}" if name else attr)
    return repaired


def _retie_shared_embeddings(model: nn.Module) -> bool:
    """Re-establish a tied output embedding after materialization.

    A tied checkpoint stores only the input embedding, so an ``lm_head`` that
    was on meta cannot be loaded from it (issue #4597). The lazy shell also
    replaces parameters as it materializes, which breaks the tie the config
    declares. Re-tie before judging whether a module is still meta.

    The load-time snapshot is authoritative: it already resolved tying from the
    wrapper flag and the checkpoint, so a wrapper whose decoder text config
    defaults to ``False`` still ties here (#4973), while a distinct checkpoint
    head stays untied (#4799).
    """

    tie = getattr(model, "tie_weights", None)
    if not callable(tie):
        return False
    snapshot = tied_weights_keys(model)
    if snapshot is not None:
        # An empty snapshot means the checkpoint really leaves the head untied.
        if not snapshot:
            return False
        return retie_tied_weights(model)
    # No snapshot (a model loaded outside the lazy path): keep the config gate.
    config = getattr(model, "config", None)
    if not word_embeddings_tied(config):
        return False
    return retie_tied_weights(model)


def shared_meta_parameter_owners(model: nn.Module) -> dict[int, list[tuple[nn.Module, str]]]:
    """Map each meta parameter owned by more than one module to its owners.

    A parameter shared by identity (an ``lm_head`` tied to the input embedding
    in session) is one tensor. Materializing each owner from its own checkpoint
    key would silently split it into two independent tensors.
    """

    owners: dict[int, list[tuple[nn.Module, str]]] = {}
    for submodule in model.modules():
        for attr, param in submodule.named_parameters(recurse=False):
            if param.is_meta:
                owners.setdefault(id(param), []).append((submodule, attr))
    return {key: value for key, value in owners.items() if len(value) > 1}


def shared_meta_parameter_snapshot(
    module: nn.Module,
    shared: dict[int, list[tuple[nn.Module, str]]],
) -> list[tuple[nn.Module, str, nn.Parameter]]:
    """Record the shared meta parameters ``module`` (recursively) is about to materialize."""

    if not shared:
        return []
    return [
        (submodule, attr, param)
        for submodule in module.modules()
        for attr, param in submodule.named_parameters(recurse=False)
        if id(param) in shared
    ]


def rebind_shared_meta_parameters(
    snapshot: list[tuple[nn.Module, str, nn.Parameter]],
    shared: dict[int, list[tuple[nn.Module, str]]],
) -> None:
    """Point every other owner of a just-materialized shared parameter at the same tensor.

    The owners that are still on meta would otherwise be filled from their own
    checkpoint keys later, breaking the identity the caller established.
    """

    for submodule, attr, old in snapshot:
        fresh = getattr(submodule, attr, None)
        if fresh is old or not isinstance(fresh, nn.Parameter) or fresh.is_meta:
            continue
        for owner, owner_attr in shared[id(old)]:
            if getattr(owner, owner_attr, None) is old:
                setattr(owner, owner_attr, fresh)


def materialize_remaining_meta_modules(qmodel, *, device: torch.device = CPU) -> List[str]:
    """Fill non-quantized modules that the lazy shell never materialized.

    With ``offload_to_disk=True`` the model is loaded as a checkpoint-backed
    meta shell, and modules outside the layer loop and the base-module list
    (an untied ``lm_head``, for example) stay on meta until ``save()`` reads
    them. ``offload_to_disk=False`` loads them into memory up front, so copy
    them from the source checkpoint here to give the in-session model the same
    end state (issue #4450). Packed quantized modules are skipped: their
    payload never comes from the dense checkpoint.

    A tied output embedding is re-tied after materialization instead of being
    read from the checkpoint, which never stores it (issue #4597).

    Raises ``RuntimeError`` when a non-quantized module still owns meta
    tensors: the in-session model would fail on its first forward or ``.to()``,
    and a partial ``.to()`` can escalate to a CUDA device-side assert, so a
    loud failure is safer than the previous log-only path.
    """

    model = qmodel.model
    materialize = getattr(qmodel, "shell_direct_meta_materialize", None)
    shared_meta = shared_meta_parameter_owners(model)
    quantized_prefixes: List[str] = []
    unresolved: List[str] = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in quantized_prefixes):
            continue
        if getattr(module, "QUANT_TYPE", None) is not None:
            quantized_prefixes.append(f"{name}.")
            continue
        if not _owns_meta_tensor(module):
            continue
        if callable(materialize):
            try:
                snapshot = shared_meta_parameter_snapshot(module, shared_meta)
                materialize(target_submodule=module, device=device)
                replace_inference_parameters(module)
                rebind_shared_meta_parameters(snapshot, shared_meta)
            except Exception as exc:  # pragma: no cover - defensive
                log.warn("Offload: failed to materialize `%s` from the source checkpoint: %s", name, exc)
        if _owns_meta_tensor(module):
            unresolved.append(name)

    # The tie must be restored before the remaining-meta verdict: a tied
    # `lm_head` is not a checkpoint tensor (issue #4597).
    _retie_shared_embeddings(model)
    unresolved = [
        name for name in unresolved if _owns_meta_tensor(model.get_submodule(name))
    ]

    if unresolved:
        raise RuntimeError(
            "Offload: modules remain on the meta device after quantization ("
            + ", ".join(unresolved[:8])
            + (", ..." if len(unresolved) > 8 else "")
            + "). The in-session model cannot run inference or be moved with "
            "`.to()`; reload the saved checkpoint before using it."
        )
    return unresolved


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


def _module_has_tensors(module: nn.Module) -> bool:
    """True when ``module`` owns state at any depth.

    ``offload_to_disk`` bundles ``module.state_dict()``, which is recursive, so
    the "is there anything to offload" gate must be recursive too. Checking
    ``recurse=False`` made container modules (``model.visual`` and friends)
    look empty and silently skipped their offload (#5145).
    """

    for parameter in module.parameters(recurse=True):
        if parameter.numel() > 0:
            return True
    for buffer in module.buffers(recurse=True):
        if buffer.numel() > 0:
            return True
    return False


def _write_offload_index(index: dict, offload_dir: str) -> None:
    """Write an Accelerate-compatible ``index.json`` into ``offload_dir``."""

    index_path = os.path.join(offload_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2)


def _bundle_module_state_dict(
    module: nn.Module,
    offload_dir: str,
    index_dir: Optional[str] = None,
    *,
    reference_index: Optional[dict] = None,
) -> dict:
    """Serialize module state into offload_dir. index_dir is the directory
    the index.json should reference -- pass the final destination when writing
    into a temporary directory that will be atomically renamed into place.

    ``reference_index`` publishes a pre-validated index that points at the
    source checkpoint instead of writing ``module.safetensors``.
    """
    if reference_index is not None:
        _write_offload_index(reference_index, offload_dir)
        return reference_index

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

    _write_offload_index(index, offload_dir)
    return index


def _read_safetensors_header(path: str) -> tuple[int, dict]:
    """Return ``(data_base, header)`` for one safetensors file."""

    size = os.path.getsize(path)
    if size < 8:
        raise ValueError(f"safetensors file is too short: {path}")
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors file has no complete header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > size - 8:
            raise ValueError(f"safetensors file has an invalid header size: {path}")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"safetensors file has invalid JSON: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return 8 + header_size, header


def _source_checkpoint_tensor_map(
    source_root: Optional[str],
    source_checkpoint: Optional[str],
) -> Optional[dict[str, tuple[str, int, dict]]]:
    """Map checkpoint tensor names to their source safetensors payload.

    Only safetensors checkpoints qualify. The returned entry is
    ``(absolute_file, data_base, header_entry)``; the caller still validates
    dtype, shape and content against the live module before publishing a
    reference.
    """

    if source_checkpoint:
        checkpoint = os.path.abspath(os.fspath(source_checkpoint))
        root = (
            os.path.abspath(os.fspath(source_root))
            if source_root
            else os.path.dirname(checkpoint)
        )
        if checkpoint.endswith(".safetensors.index.json"):
            return _source_shard_tensor_map(root, checkpoint)
        if checkpoint.endswith(".safetensors"):
            data_base, header = _read_safetensors_header(checkpoint)
            return {
                name: (checkpoint, data_base, entry)
                for name, entry in header.items()
                if name != "__metadata__" and isinstance(entry, dict)
            }
        return None

    if not source_root:
        return None
    root = os.path.abspath(os.fspath(source_root))
    index_path = os.path.join(root, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        return _source_shard_tensor_map(root, index_path)
    single_path = os.path.join(root, "model.safetensors")
    if os.path.isfile(single_path):
        data_base, header = _read_safetensors_header(single_path)
        return {
            name: (single_path, data_base, entry)
            for name, entry in header.items()
            if name != "__metadata__" and isinstance(entry, dict)
        }
    return None


def _source_shard_tensor_map(root: str, index_path: str) -> Optional[dict[str, tuple[str, int, dict]]]:
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return None

    headers: dict[str, tuple[int, dict]] = {}
    tensor_map: dict[str, tuple[str, int, dict]] = {}
    for name, shard in weight_map.items():
        try:
            shard_path = resolve_checkpoint_path(
                root,
                shard,
                kind="checkpoint weight_map shard",
                follow_symlinks=False,
            )
        except Exception:
            return None
        if shard_path not in headers:
            try:
                headers[shard_path] = _read_safetensors_header(shard_path)
            except (OSError, ValueError):
                return None
        data_base, header = headers[shard_path]
        entry = header.get(name)
        if not isinstance(entry, dict):
            return None
        tensor_map[str(name)] = (shard_path, data_base, entry)
    return tensor_map


def _source_payload_sha256_if_equal(
    tensor: torch.Tensor,
    path: str,
    offset: int,
    nbytes: int,
) -> Optional[str]:
    """Return the payload's sha256 when ``tensor`` holds exactly those bytes.

    The source range is streamed through one reusable buffer of at most
    ``_SOURCE_COMPARE_CHUNK_BYTES``, and the live tensor is compared through a
    byte view of its own storage. Neither side is materialized as a second
    full-size host tensor, and the source is read with ``readinto`` rather than
    mapped, so the check adds at most one chunk to the process RSS. The digest
    is computed over the bytes just compared, which are the live tensor's
    bytes. Returns ``None`` on any byte difference or a short source.
    """

    live = tensor.detach()
    if live.is_conj() or live.is_neg():
        live = live.resolve_conj().resolve_neg()
    live = live.reshape(-1).view(torch.uint8)
    if live.numel() != nbytes:
        return None
    digest = hashlib.sha256()
    if nbytes == 0:
        return digest.hexdigest()

    scratch = bytearray(min(nbytes, _SOURCE_COMPARE_CHUNK_BYTES))
    scratch_view = memoryview(scratch)
    scratch_tensor = torch.frombuffer(scratch, dtype=torch.uint8)
    with open(path, "rb", buffering=0) as handle:
        handle.seek(offset)
        position = 0
        while position < nbytes:
            length = min(len(scratch), nbytes - position)
            filled = 0
            while filled < length:
                read = handle.readinto(scratch_view[filled:length])
                if not read:
                    return None
                filled += read
            live_chunk = live[position:position + length]
            if live_chunk.device.type != "cpu":
                live_chunk = live_chunk.to("cpu")
            if not torch.equal(scratch_tensor[:length], live_chunk):
                return None
            digest.update(scratch_view[:length])
            position += length
    return digest.hexdigest()


def _module_source_reference_index(
    module: nn.Module,
    *,
    module_name: str,
    source_root: Optional[str],
    source_checkpoint: Optional[str],
    offload_dir: str,
) -> Optional[dict]:
    """Build an Accelerate index that points at untouched checkpoint tensors.

    Returns ``None`` when any module tensor is absent from the source, has a
    different dtype/shape, differs in content, or lives inside the offload
    directory. The caller then falls back to serializing a bundle. Content is
    proven byte for byte against the source payload in bounded chunks
    (`_source_payload_sha256_if_equal`), never by loading the source tensor.
    """

    try:
        tensor_map = _source_checkpoint_tensor_map(source_root, source_checkpoint)
    except (OSError, ValueError):
        return None
    if not tensor_map:
        return None

    state = module.state_dict()
    if not state:
        return None
    full_name = module_name
    offload_real = os.path.realpath(offload_dir)
    index: dict[str, dict] = {}

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            return None
        source_key = f"{full_name}.{key}" if full_name else key
        found = tensor_map.get(source_key)
        if found is None:
            return None
        path, data_base, entry = found
        try:
            if os.path.commonpath((os.path.realpath(path), offload_real)) == offload_real:
                return None
        except ValueError:
            # Paths on different drives (Windows) have no common prefix, so the
            # source cannot be inside the offload dir; keep the reference.
            pass

        try:
            expected_dtype = _safetensors_dtype(tensor.dtype)
        except ValueError:
            return None
        offsets = entry.get("data_offsets")
        if (
            entry.get("dtype") != expected_dtype
            or entry.get("shape") != list(tensor.shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            return None
        start, end = int(offsets[0]), int(offsets[1])
        if end - start != tensor.numel() * tensor.element_size():
            return None

        try:
            sha256 = _source_payload_sha256_if_equal(tensor, path, data_base + start, end - start)
        except OSError:
            return None
        if sha256 is None:
            return None

        index[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "safetensors_file": path,
            "weight_name": source_key,
            "data_offsets": [data_base + start, data_base + end],
            "sha256": sha256,
        }
    return index


def _qzero_layout_for_module(module: nn.Module) -> Optional[int]:
    """Runtime qzero layout the restored module decodes, or ``None``.

    Only GPTQ-style modules expose the layout contract; AWQ words have no
    v1/v2 convention, so they are left untouched.
    """

    target = getattr(module, "_load_target_qzeros_format", None)
    if not callable(target):
        return None
    return int(target())


def _bundle_qzero_layout(module: nn.Module, weights_map: Mapping) -> Optional[int]:
    """Layout of the ``qzeros`` words stored in an offload bundle.

    The bundle is written from ``module.state_dict()``, so it holds the
    checkpoint layout and its ``qzeros_format`` marker. A bundle written before
    the marker existed falls back to the module's recorded checkpoint format,
    which is the same fallback the checkpoint load hook uses.
    """

    try:
        marker = weights_map[_qzeros_format_buffer()]
    except (KeyError, TypeError, IndexError):
        marker = None
    if isinstance(marker, torch.Tensor) and marker.numel() == 1:
        code = int(marker.reshape(-1)[0].item())
        return code if code in (1, 2) else None

    incoming = getattr(module, "_incoming_qzeros_format", None)
    if callable(incoming):
        code = incoming({})
        if code in (1, 2):
            return code
    return None


def _qzeros_format_buffer() -> str:
    from ..nn_modules.qlinear._fused_train_state import QZEROS_FORMAT_BUFFER

    return QZEROS_FORMAT_BUFFER


def _convert_bundle_qzeros(
    module: nn.Module,
    qzeros: torch.Tensor,
    source: int,
    target: int,
) -> torch.Tensor:
    """Return ``qzeros`` converted from the bundle layout to the live layout."""

    if not isinstance(qzeros, torch.Tensor) or qzeros.numel() == 0:
        return qzeros
    bits = getattr(module, "bits", None)
    if bits is None:
        return qzeros
    planar = bool(getattr(module, "planar", False))
    if source == 1 and target == 2:
        from .model_dequant import _correct_gptq_v1_qzeros

        return _correct_gptq_v1_qzeros(qzeros, bits, planar=planar)
    if source == 2 and target == 1:
        from .model_dequant import _revert_gptq_v1_qzeros_correction

        return _revert_gptq_v1_qzeros_correction(qzeros, bits, planar=planar)
    return qzeros


class _LiveQzeroWeightsMap(Mapping):
    """Restore ``qzeros`` in the live runtime layout from a disk bundle.

    ``accelerate.disk_offload`` copies the bundle tensors straight into the
    module; it never runs ``_load_from_state_dict``. A bundle written from
    ``state_dict()`` therefore restores checkpoint-layout (v1) words into a
    kernel that decodes v2, shifting every zero point by one (issue #4380).
    Converting at the weights-map boundary fixes every restore consumer -- the
    forward pre-hook, hook detach, and ``undo_offload_to_disk`` -- while the
    on-disk bundle stays in checkpoint layout for saving.
    """

    def __init__(self, base: Mapping, module: nn.Module):
        self._base = base
        self._module = module
        self._source = _bundle_qzero_layout(module, base)
        self._target = _qzero_layout_for_module(module)

    def __getitem__(self, key):
        value = self._base[key]
        if (
            self._source is None
            or self._target is None
            or self._source == self._target
        ):
            return value
        if key == "qzeros" or key.endswith(".qzeros"):
            return _convert_bundle_qzeros(
                self._module, value, self._source, self._target,
            )
        if key == _qzeros_format_buffer() or key.endswith(
            f".{_qzeros_format_buffer()}"
        ):
            # Keep the restored marker consistent with the converted words so a
            # later clone/load of the live buffers does not relabel them as v1.
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                return value.clone().fill_(self._target)
        return value

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _iter_offload_hooks(module: nn.Module):
    """Yield ``(owning_module, AlignDevicesHook)`` for installed offload hooks."""

    for submodule in module.modules():
        pending = [getattr(submodule, "_hf_hook", None)]
        while pending:
            hook = pending.pop()
            if hook is None:
                continue
            nested = getattr(hook, "hooks", None)
            if isinstance(nested, (tuple, list)):
                pending.extend(nested)
                continue
            if getattr(hook, "offload", False):
                yield submodule, hook


def _install_live_qzero_restore(module: nn.Module) -> None:
    """Make the accelerate restore path hand the module live-layout qzeros."""

    target = _qzero_layout_for_module(module)
    if target is not None and hasattr(module, "_qzeros_format"):
        # A restore now materializes the runtime words, so the module's label
        # must describe what will be put in its buffers.
        module._qzeros_format = target
        module._qzeros_format_established = True

    for owner, hook in _iter_offload_hooks(module):
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None or isinstance(weights_map, _LiveQzeroWeightsMap):
            continue
        hook.weights_map = _LiveQzeroWeightsMap(weights_map, owner)


def _safetensors_dtype(dtype: torch.dtype) -> str:
    """Resolve the safetensors header spelling through the canonical export map.

    Do not duplicate the table here: the float8 spellings are non-obvious
    (``F8_E4M3``, not ``F8_E4M3FN``) and a second copy silently drifts. The
    import is local because ``utils/model.py`` imports this module lazily.
    """

    from .model import _torch_dtype_to_safetensors

    try:
        return _torch_dtype_to_safetensors(dtype)
    except NotImplementedError as error:
        raise ValueError(
            f"safetensors reference offload cannot serialize dtype {dtype}"
        ) from error


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

    Note: no Accelerate offload hooks are attached, so ``undo_offload_to_disk``
    will not materialize the module again. This path is intended for export via
    ``get_state_dict_for_save``; the caller keeps ownership of the source file.
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

    module_offload_dir = os.path.join(disk_path, full_name)
    source_real = os.path.realpath(source)
    offload_real = os.path.realpath(module_offload_dir)
    try:
        source_in_offload_dir = os.path.commonpath((source_real, offload_real)) == offload_real
    except ValueError:  # Different drives on Windows cannot overlap.
        source_in_offload_dir = False
    if source_in_offload_dir:
        raise ValueError(
            "safetensors reference source is inside the target offload directory, "
            "which is cleared before the index is written"
        )

    state = module.state_dict()
    if not state:
        raise ValueError("safetensors reference module has no persistent tensors")

    # ``to_empty`` replaces tensor parameters and buffers, but only persistent
    # ones are recorded in state_dict()/the index. Optional buffers set to None
    # stay None; reject only live non-persistent tensors that would be lost.
    non_persistent_buffers = sorted(
        f"{prefix}.{buffer_name}" if prefix else buffer_name
        for prefix, submodule in module.named_modules()
        for buffer_name in submodule._non_persistent_buffers_set
        if submodule._buffers.get(buffer_name) is not None
    )
    if non_persistent_buffers:
        raise ValueError(
            "safetensors reference offload cannot preserve non-persistent buffers "
            f"(e.g. `{non_persistent_buffers[0]}`)"
        )

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
        expected_dtype = _safetensors_dtype(tensor.dtype)
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
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
        _owner, buffer = tensor_buffer(tensor)
        index[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "safetensors_file": source,
            "weight_name": key,
            "data_offsets": [data_base + start, data_base + end],
            "sha256": hashlib.sha256(buffer).hexdigest(),
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

    _prepare_offload_directory(module_offload_dir)
    index_path = os.path.join(module_offload_dir, "index.json")
    temporary = f"{index_path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
        if os.name == "posix":
            directory = os.open(
                module_offload_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
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


def offload_to_disk(
    module: List[str] | nn.Module,
    model: nn.Module,
    disk_path: str = ".",
    force: bool = False,
    *,
    source_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
):
    """Offload module state to disk. force=True also writes modules below the
    small-module threshold -- required when the on-disk bundle is the only
    durable record (e.g. quantized modules that a resumed run must restore).

    ``source_root``/``source_checkpoint`` name the checkpoint the model was
    loaded from. When a module's tensors are still byte-identical to that
    checkpoint, the offload index references the source bytes instead of
    writing a second copy.
    """
    _offload_to_disk_impl(
        module=module,
        model=model,
        disk_path=disk_path,
        force=force,
        source_root=source_root,
        source_checkpoint=source_checkpoint,
    )


def _offload_to_disk_impl(
    module: List[str] | nn.Module,
    model: nn.Module,
    disk_path: str = ".",
    force: bool = False,
    *,
    source_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
):
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
            _offload_disk(
                module=m,
                name=full_name,
                disk_path=disk_path,
                force=force,
                source_root=source_root,
                source_checkpoint=source_checkpoint,
            )
    else:
        # unwrap named module
        if isinstance(module, NamedModule):
            # print(f"offloading named module: {module.full_name}")
            module = module.module

        full_name = get_module_fullname(model=model, module=module)

        _offload_disk(
            module=module,
            name=full_name,
            disk_path=disk_path,
            force=force,
            source_root=source_root,
            source_checkpoint=source_checkpoint,
        )

    if hasattr(module, "config") and hasattr(module, "tie_weights"):
        # Re-applies only the pairs the checkpoint really ties (see
        # `gptqmodel.utils.tied_weights`), so an untied `lm_head` keeps its own
        # storage after offload/restore (#4799).
        retie_tied_weights(module)

    # print("offload_disk: list item tree")
            # print_module_tree(module)


# Serialize accelerate's disk hook mutations across threads.
#_OFFLOAD_SAFE = ThreadSafe(sys.modules[__name__])
#offload_to_disk = _OFFLOAD_SAFE.offload_to_disk

def _offload_disk(
    module: nn.Module,
    name: str,
    disk_path: str = ".",
    force: bool = False,
    *,
    source_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
):
    with parent_module_lock(name):
        _offload_disk_locked(
            module=module,
            name=name,
            disk_path=disk_path,
            force=force,
            source_root=source_root,
            source_checkpoint=source_checkpoint,
        )


def _offload_disk_locked(
    module: nn.Module,
    name: str,
    disk_path: str = ".",
    force: bool = False,
    *,
    source_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
):
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

    # Skip only a module whose recursive state is empty. The old check used
    # recurse=False, so a container that owns no direct tensors (a multimodal
    # ``model.visual`` tower, whose weights all live in its children) was
    # skipped as "empty" and silently stayed on the accelerator for the whole
    # layer loop (#5145). ``state_dict()`` below is recursive, so this check
    # now matches what actually gets bundled.
    if not _module_has_tensors(module):
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
    reference_index = None
    if source_root or source_checkpoint:
        try:
            reference_index = _module_source_reference_index(
                module,
                module_name=name,
                source_root=source_root,
                source_checkpoint=source_checkpoint,
                offload_dir=module_offload_dir,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Offload: source reference unavailable for `%s`: %s", name, exc)
    _bundle_module_state_dict(
        module,
        tmp_dir,
        index_dir=module_offload_dir,
        reference_index=reference_index,
    )

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
    # The bundle is checkpoint layout (v1 words for a GPTQ checkpoint), but the
    # accelerate hook loads it without the module's load hook. Hand the hook
    # live-layout qzeros so a v2 kernel does not decode every zero point off by
    # one in-session (issue #4380).
    _install_live_qzero_restore(module)

    # print("offload_disk: list item tree")
    # print_module_tree(module)


def _materialize_meta_nonpersistent_buffers(module: nn.Module) -> List[str]:
    """Replace non-persistent buffers that are still on meta with real tensors.

    ``disk_offload`` builds its index from ``state_dict()``, which excludes
    non-persistent buffers by definition. A module rebuilt on ``meta`` for a
    checkpoint resume (#4562) therefore keeps, for example,
    ``QQQLinear.workspace`` on meta with no stored value, and accelerate's hook
    init raises ``ValueError: workspace is on the meta device`` (#4777). Those
    buffers are runtime scratch and never checkpoint state, so allocate them
    here instead of trying to load them. Returns the qualified names that were
    materialized.
    """

    materialized: List[str] = []
    for prefix, submodule in module.named_modules():
        for name, tensor in list(submodule._buffers.items()):
            if tensor is None or not tensor.is_meta:
                continue
            if name not in submodule._non_persistent_buffers_set:
                # Persistent: the offload index carries it and the hook loads it.
                continue
            submodule._buffers[name] = torch.zeros(
                tensor.shape, dtype=tensor.dtype, device=CPU
            )
            materialized.append(f"{prefix}.{name}" if prefix else name)
    return materialized


# undo offload
def attach_disk_offload_hook(
    module: nn.Module,
    offload_dir: str,
    *,
    execution_device: torch.device = CPU,
) -> None:
    """Register an already-meta module with an existing offload bundle.

    A resumed run restores completed packed modules as meta tensors from the
    checkpoint object and writes an Accelerate-format ``index.json`` for them
    (issue #4562). Without a hook the loop-exit restore never sees those
    modules, so they stay on meta and the in-session forward fails silently.
    Attaching the same hook ``_offload_disk_locked`` installs lets
    :func:`restore_disk_offloaded_modules` materialize them at loop exit and
    raise the reload-the-checkpoint error when the bundle is unreadable.

    ``AlignDevicesHook.detach_hook`` only reloads leaves whose recorded
    ``original_devices`` entry is not meta. A module that was already meta when
    the hook attached has no such entry, so its recorded devices are rewritten
    to ``execution_device``; otherwise detaching would leave every tensor meta.

    ``offload_dir`` must already contain the index; a missing one is an error,
    not something to regenerate from meta tensors.
    """

    if not os.path.isfile(os.path.join(offload_dir, "index.json")):
        raise RuntimeError(
            f"Offload: no offload index for `{offload_dir}`; the resumed packed "
            "module cannot be materialized after quantization"
        )
    # The index is built from ``state_dict()``, so a non-persistent buffer (for
    # example QQQ's split-K ``workspace``) has no stored value and would still
    # be meta when accelerate initializes the hook (#4777).
    materialized = _materialize_meta_nonpersistent_buffers(module)
    if materialized:
        log.debug(
            "Offload: materialized non-persistent buffers for resumed module: %s",
            ", ".join(materialized),
        )
    disk_offload(
        module,
        offload_dir=offload_dir,
        offload_buffers=True,
        execution_device=execution_device,
    )
    _install_live_qzero_restore(module)
    for _owner, hook in _iter_offload_hooks(module):
        devices = getattr(hook, "original_devices", None)
        if not isinstance(devices, dict):
            continue
        hook.original_devices = {
            name: (execution_device if torch.device(device).type == "meta" else device)
            for name, device in devices.items()
        }


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


def _drop_restored_class_forward(module: nn.Module) -> None:
    """Return ``forward`` dispatch to the module's class after hook removal.

    ``remove_hook_from_module`` restores the pre-hook forward by assigning the
    bound method to the instance (``module.forward = module._old_forward``).
    That pins the class the module had when it was offloaded: a later
    in-place kernel switch (``rehome_quantized_kernel`` swapping AWQ GEMM for
    the portable Torch kernel) would keep running the old class's forward.
    Drop the instance attribute when it is just the class's own forward.
    """

    bound = module.__dict__.get("forward")
    if (
        bound is not None
        and getattr(bound, "__self__", None) is module
        and getattr(bound, "__func__", None) is getattr(type(module), "forward", None)
    ):
        del module.__dict__["forward"]


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
            for sub in module.modules():
                _drop_restored_class_forward(sub)

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
            tie_weights = getattr(module, "tie_weights", None)
            if callable(tie_weights) and hasattr(module, "config"):
                # Submodules such as rotary embeddings expose `config` but are
                # not models; `retie_tied_weights` is a no-op without a snapshot
                # when the config declares untied embeddings, and re-applies the
                # resolved pairs otherwise (#4799).
                retie_tied_weights(module)  # makes a tied lm_head point to embed_tokens.weight again

        # 4) Optionally delete offload folders.
        if delete_offload_folders:
            for d in sorted(offload_dirs):
                with contextlib.suppress(Exception):
                    shutil.rmtree(d, ignore_errors=True)

        return module
