# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Restore packed checkpoint state when a fused CPU layer resumes training."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
import torch.nn as nn


_STATE_NAMES = ("qweight", "qzeros", "scales")
_STATE_BUFFER_PREFIX = "_fused_train_state_"


@dataclass(frozen=True)
class RebuiltCheckpointTensor:
    """Checkpoint operand that is rebuilt from the live runtime layout.

    ``qweight`` is not stored twice. The CPU int4pack permutation is a pure
    nibble permutation whose inverse :func:`finalize_fused_train_state` verified
    against this module's own checkpoint bytes, so the checkpoint words are
    reconstructed on ``state_dict()``/save and when training restores the
    module. Only the checkpoint shape and dtype stay resident.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class RebuildCandidate:
    """Checkpoint clone kept only until the transform proves its inverse.

    ``capture_fused_train_state`` still clones ``qweight`` for a module whose
    class declares an invertible int4pack layout. The transform then calls
    :func:`finalize_fused_train_state`, which rebuilds the checkpoint words from
    the packed tensor and compares them with this clone. A match replaces the
    clone with :class:`RebuiltCheckpointTensor`; any mismatch keeps the clone,
    so an unverified inverse can never reach a saved checkpoint.
    """

    buffer_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


#: Entries that describe the checkpoint layout instead of naming a buffer.
_DESCRIBED_ENTRIES = (RebuiltCheckpointTensor, RebuildCandidate)

#: Layouts whose rebuilt checkpoint bytes were already proven bitwise
#: identical to a captured clone in this process. The first module of each
#: key pays one full rebuild; every later module of the same layout only
#: checks the captured shape/dtype. This is what keeps the ``#4783`` memory
#: win without making every ``state_dict()``/first forward rebuild every
#: weight (issue #4884).
_VERIFIED_REBUILD_LAYOUTS: set[tuple] = set()
#: Free-threaded builds can finalize several layers at once; the set only
#: guards membership and insertion, never the rebuild itself.
_VERIFIED_REBUILD_LAYOUTS_LOCK = threading.Lock()


def _rebuild_layout_key(module: nn.Module, rebuild) -> tuple:
    """Identify a module layout whose inverse has been proven in this process.

    The key carries the inverse implementation itself, so a subclass override
    or a runtime monkeypatch re-verifies instead of inheriting an unrelated
    proof. The forward packer is part of the key too: the proof says that this
    inverse reproduces *that* packer, so a different kernel must re-verify.
    Shapes and the ``ret_idx`` rebuild mode are included because they select
    different inverse branches.
    """

    function = getattr(rebuild, "__func__", rebuild)
    parts: list = [type(module), function]
    for name in (
        "_FUSED_TRAIN_STATE_REBUILD",
        "_cpu_runtime_layout",
        "bits",
        "group_size",
        "pack_factor",
        "out_features",
        "in_features",
    ):
        parts.append(getattr(module, name, None))
    for name in ("pack_gptq_int4_kernel", "pack_awq_int4_kernel"):
        parts.append(getattr(module, name, None))
    if getattr(module, "_FUSED_TRAIN_STATE_REBUILD", None) in ("gptq", "awq"):
        from ._cpu_int4pack_layout import int4pack_block_size

        parts.append(int4pack_block_size())
    if getattr(module, "_FUSED_TRAIN_STATE_REBUILD", None) == "avx512_awq":
        # ``pack_awq_int4`` applies no channel reorder, so every module of this
        # layout shares one inverse and one proof (#5013).
        parts.append("awq-noreorder")
        return tuple(parts)
    identity = getattr(module, "_ret_idx_is_identity", None)
    if not callable(identity):
        parts.append(None)
        return tuple(parts)
    try:
        if identity():
            parts.append("identity")
        elif getattr(module, "_FUSED_TRAIN_STATE_REBUILD", None) == "avx512_gptq":
            from ._cpu_int4pack_layout import avx512_checkpoint_byte_index

            parts.append(
                "bytes" if avx512_checkpoint_byte_index(module) is not None else "codes"
            )
        else:
            parts.append("reordered")
    except Exception:
        parts.append(None)
    return tuple(parts)


def _entry_shape(entry):
    """Checkpoint shape an entry describes, or ``None`` for a buffer name."""

    if isinstance(entry, _DESCRIBED_ENTRIES):
        return entry.shape
    return None


def _entry_buffer_name(entry):
    """Resident snapshot buffer backing an entry, or ``None`` if it has none."""

    if isinstance(entry, RebuildCandidate):
        return entry.buffer_name
    if isinstance(entry, str):
        return entry
    return None


def _materialize_snapshot_entry(module: nn.Module, entry):
    """Return the checkpoint tensor described by a snapshot entry."""

    if isinstance(entry, RebuiltCheckpointTensor):
        tensor = module._rebuild_checkpoint_qweight()
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"{module.__class__.__name__} could not rebuild its checkpoint qweight"
            )
        if tuple(tensor.shape) != entry.shape or tensor.dtype != entry.dtype:
            raise RuntimeError(
                f"{module.__class__.__name__} rebuilt qweight with shape/dtype "
                f"{tuple(tensor.shape)}/{tensor.dtype}, expected "
                f"{entry.shape}/{entry.dtype}"
            )
        return tensor
    buffer_name = _entry_buffer_name(entry)
    if buffer_name is None:
        return None
    return module._buffers.get(buffer_name)

#: Persistent buffer that records the qzero word layout a state dict carries.
#:
#: ``state_dict._metadata`` is not a tensor, so a safetensors round trip
#: (Trainer/accelerate) or a plain ``{k: v for k, v in sd.items()}`` copy drops
#: it. The layout therefore has to travel inside the payload. Values:
#: ``1`` = v1 checkpoint words, ``2`` = v2 checkpoint words, ``0`` = the tensors
#: are a kernel-private runtime layout and not a checkpoint.
QZEROS_FORMAT_BUFFER = "qzeros_format"


def capture_fused_train_state(module: nn.Module, *, names=None) -> None:
    """Retain packed checkpoint tensors before an inference-only transform.

    The clone deliberately escapes the caller's ``inference_mode``. Caches are
    captured from inside an inference forward, but the snapshot is later
    restored as the checkpoint tensor: ``load_state_dict`` copies into it in
    place outside ``inference_mode``, which Torch rejects for inference tensors.

    A kernel whose CPU transform leaves an operand byte-identical to the live
    tensor does not need a second copy of it. Such a kernel declares its
    snapshot names through ``_fused_train_state_names`` (or
    ``_FUSED_TRAIN_STATE_NAMES``), and the untouched operand is emitted
    directly from the live buffer by ``state_dict()``.

    ``qweight`` is cloned only as a :class:`RebuildCandidate` when the module
    declares an invertible int4pack layout. The clone is dropped by
    :func:`finalize_fused_train_state` once the transform proves that the
    checkpoint words can be rebuilt from the packed tensor, which removes the
    largest resident duplicate while leaving the emitted checkpoint
    byte-identical.
    """

    if getattr(module, "_fused_train_state", None) is not None:
        return
    if names is None:
        resolver = getattr(module, "_fused_train_state_names", None)
        if callable(resolver):
            names = resolver()
        else:
            names = getattr(module, "_FUSED_TRAIN_STATE_NAMES", _STATE_NAMES)
    rebuild_qweight = getattr(module, "_can_rebuild_checkpoint_qweight", None)
    snapshot = {}
    with torch.inference_mode(False):
        for name in names:
            tensor = getattr(module, name, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            buffer_name = f"{_STATE_BUFFER_PREFIX}{name}"
            module.register_buffer(
                buffer_name,
                tensor.detach().clone(),
                persistent=False,
            )
            if (
                name == "qweight"
                and callable(rebuild_qweight)
                and rebuild_qweight()
            ):
                snapshot[name] = RebuildCandidate(
                    buffer_name, tuple(tensor.shape), tensor.dtype
                )
            else:
                snapshot[name] = buffer_name
        module._fused_train_state = snapshot
        module._fused_train_state_exportable = True


def finalize_fused_train_state(module: nn.Module) -> bool:
    """Drop the ``qweight`` clone after proving its inverse on this module.

    Called at the end of a CPU int4pack transform, once the live ``qweight``
    holds the packed words. The checkpoint words are rebuilt from that tensor
    and compared bitwise with the clone captured before the transform. Only an
    exact match removes the clone and keeps a :class:`RebuiltCheckpointTensor`
    descriptor; a layout this build cannot invert (a different PyTorch packing,
    a fused operand order, an unsupported dtype) keeps the resident clone, so
    the exported checkpoint stays byte-identical either way.

    The comparison runs once per layout in a process: the inverse is a
    deterministic function of the layout, so a later module of the same layout
    only needs the captured shape/dtype. Without that cache every module
    rebuilds its whole weight here (issue #4884).

    Returns whether any clone was dropped.
    """

    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot:
        return False
    rebuild = getattr(module, "_rebuild_checkpoint_qweight", None)
    if not callable(rebuild):
        return False
    dropped = False
    for name, entry in tuple(snapshot.items()):
        if not isinstance(entry, RebuildCandidate):
            continue
        expected = module._buffers.get(entry.buffer_name)
        if not isinstance(expected, torch.Tensor):
            continue
        descriptor = RebuiltCheckpointTensor(entry.shape, entry.dtype)
        key = _rebuild_layout_key(module, rebuild)
        with _VERIFIED_REBUILD_LAYOUTS_LOCK:
            verified = key in _VERIFIED_REBUILD_LAYOUTS
        if not verified:
            try:
                rebuilt = _materialize_snapshot_entry(module, descriptor)
            except Exception:
                # Fail closed: an unverified inverse keeps the resident clone.
                continue
            if rebuilt is None or not torch.equal(rebuilt, expected):
                continue
            with _VERIFIED_REBUILD_LAYOUTS_LOCK:
                _VERIFIED_REBUILD_LAYOUTS.add(key)
        # Publish the descriptor before dropping the clone so a reader that
        # resolves the snapshot entry never sees a name whose buffer is gone
        # (issue #4788). `state_dict()` also holds `_cpu_transform_lock`, which
        # is what makes the pair atomic.
        snapshot[name] = descriptor
        module._buffers.pop(entry.buffer_name, None)
        dropped = True
    return dropped


def bind_runtime_qzeros(module: nn.Module) -> bool:
    """Point ``qzeros`` at the zero column of the packed affine metadata.

    The CPU int4pack transform stores ``scales_and_zeros`` as
    ``[out_features, groups, 2]`` with the zero offsets in the second column and
    then keeps a standalone ``qzeros`` tensor holding the same offsets. Reusing
    that column removes the duplicate storage without changing any value: the
    affine metadata already owns those bytes. Returns whether ``qzeros`` is now
    a view of ``scales_and_zeros``.

    A fused source owns no storage of its own: ``_share_operands`` rebinds its
    operands to views of the fused allocation, so re-pointing ``qzeros`` at a
    private ``scales_and_zeros`` would break that alias and reintroduce a copy.
    """

    if getattr(module, "_fused_operands_shared", False):
        return False
    packed = getattr(module, "scales_and_zeros", None)
    if not isinstance(packed, torch.Tensor):
        return False
    if packed.dim() != 3 or tuple(packed.shape[-1:]) != (2,):
        return False
    module.qzeros = packed[..., 1]
    return True


def _replace_tensor(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    current = getattr(module, name, None)
    if isinstance(current, nn.Parameter):
        module._parameters[name] = nn.Parameter(
            tensor,
            requires_grad=current.requires_grad,
        )
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


def restore_fused_train_state(module: nn.Module) -> bool:
    """Restore checkpoint-packed tensors captured before a fused transform.

    The live operands return to the checkpoint layout, so any state that
    describes the inference-only runtime layout is discarded as well. The next
    forward transforms the module again.
    """

    if getattr(module, "_suspend_fused_train_restore", False):
        return False
    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot:
        return False
    if not getattr(module, "_fused_train_state_exportable", True):
        raise RuntimeError(
            "Cannot restore or save a modified CPU int4pack runtime state as a checkpoint. "
            "Load checkpoint-layout weights to restore the checkpoint format."
        )
    # Rebuild every derived operand before replacing anything: a rebuild reads
    # the live runtime tensor it is derived from.
    rebuilt = {}
    for name, entry in snapshot.items():
        if isinstance(entry, RebuiltCheckpointTensor):
            rebuilt[name] = _materialize_snapshot_entry(module, entry)
    for name, entry in snapshot.items():
        if isinstance(entry, RebuiltCheckpointTensor):
            _replace_tensor(module, name, rebuilt[name])
            continue
        tensor = module._buffers.pop(_entry_buffer_name(entry), None)
        if tensor is not None:
            _replace_tensor(module, name, tensor)
    discard_fused_train_state(module)
    reset_fused_runtime_state(module)
    return True


def fused_autograd_needs_dense(module: nn.Module, x: torch.Tensor) -> bool:
    """Whether a latched fused inference path must yield to autograd.

    The fused CPU kernels run under ``no_grad``. Taking one while the caller
    needs a gradient silently detaches the output (#5314), so the forward must
    use the differentiable dequant path for that call instead.
    """

    if not (torch.is_grad_enabled() and x.requires_grad):
        return False
    mode = getattr(module, "linear_mode", None)
    return getattr(mode, "value", mode) == "inference"


def reset_fused_runtime_state(module: nn.Module) -> None:
    """Forget that the live operands are in a kernel's runtime layout.

    This only drops derived state. The packed tensors themselves are left
    untouched; a later checkpoint load or explicit restore decides their
    layout.
    """

    for name, value in (
        ("linear_mode", None),
        ("torch_fused_op", None),
        ("scales_and_zeros", None),
        ("_cpu_transformed", False),
        ("_cpu_transform_dtype", None),
        ("_fused_state_key", None),
    ):
        if hasattr(module, name):
            setattr(module, name, value)


def refinalize_fused_state_after_apply(
    module: nn.Module,
    previous_scale_dtype,
    previous_device,
) -> bool:
    """Re-finalize the checkpoint layout after a move invalidates the runtime one.

    A runtime int4pack layout is only valid for the device that produced it:
    the CPU packing is not the XPU packing, so a cross-device move must
    re-transform instead of handing one backend's packed weight to another.
    The runtime affine offsets are also computed from the scale dtype, so
    narrowing ``scales``/``qzeros`` in place would pair the new scales with
    offsets built from the old precision and silently change the decoded
    weights. A same-device move to FP32 only widens those offsets, so it keeps
    the live layout.

    A modified runtime state has no valid checkpoint layout to re-finalize
    from, so ``restore_fused_train_state`` raises the same error as export.
    """

    if previous_scale_dtype is None:
        return False
    scales = getattr(module, "scales", None)
    if not isinstance(scales, torch.Tensor):
        return False
    if scales.device != previous_device:
        return restore_fused_train_state(module)
    if scales.dtype == previous_scale_dtype or scales.dtype not in (torch.float16, torch.bfloat16):
        return False
    return restore_fused_train_state(module)


def checkpoint_layout_tensor(module: nn.Module, name: str):
    """Return the checkpoint tensor a CPU transform replaced with a runtime one.

    Returns ``None`` when no transform replaced ``name``. A transformed module
    whose runtime operands were modified has no guaranteed inverse, so callers
    that serialize or repack must fail closed instead of copying the runtime
    ABI. The snapshot is the canonical checkpoint payload; using it here keeps
    a repack independent of whether a forward already ran.
    """

    snapshot = getattr(module, "_fused_train_state", None)
    if not snapshot or name not in snapshot:
        return None
    if not getattr(module, "_fused_train_state_exportable", True):
        raise RuntimeError(
            f"{type(module).__name__} holds a modified CPU int4pack runtime state "
            "that cannot be converted back to the checkpoint layout. Load "
            "checkpoint-layout weights before saving or repacking."
        )
    tensor = _materialize_snapshot_entry(module, snapshot[name])
    return tensor if isinstance(tensor, torch.Tensor) else None


def discard_fused_train_state(module: nn.Module) -> None:
    snapshot = getattr(module, "_fused_train_state", None)
    if snapshot:
        for entry in snapshot.values():
            buffer_name = _entry_buffer_name(entry)
            if buffer_name is not None:
                module._buffers.pop(buffer_name, None)
    module._fused_train_state = None
    module.__dict__.pop("_fused_train_state_exportable", None)


def runtime_state_matches_live(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether a runtime-layout load leaves every local checkpoint operand unchanged."""

    for name, live in (*module._parameters.items(), *module._buffers.items()):
        if name in module._non_persistent_buffers_set:
            continue
        if name == QZEROS_FORMAT_BUFFER:
            # Serialization metadata, not a packed operand.
            continue
        incoming = state_dict.get(f"{prefix}{name}")
        if incoming is None:
            continue
        if not isinstance(live, torch.Tensor) or not isinstance(incoming, torch.Tensor):
            return False
        if live.dtype != incoming.dtype or live.shape != incoming.shape or not torch.equal(live, incoming):
            return False
    return True


def materialize_fused_runtime_state(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Replace inference-tensor operands with normal tensors in place.

    The CPU inference transform runs inside ``inference_mode``, so the live
    checkpoint operands, including ``g_idx``, can be inference tensors.
    ``load_state_dict(assign=False)`` copies into them outside
    inference mode, which Torch rejects. Cloning them into normal tensors keeps
    the current layout while allowing that copy; ``assign=True`` does not need
    it because the load replaces the tensors outright. ``g_idx`` is shared by
    the checkpoint and runtime layouts and is never part of the snapshot, so
    this applies to checkpoint loads as well.
    """

    materialized = False
    with torch.inference_mode(False):
        for name in (*module._parameters, *module._buffers):
            if f"{prefix}{name}" not in state_dict:
                continue
            live = getattr(module, name, None)
            if isinstance(live, torch.Tensor) and torch.is_inference(live):
                _replace_tensor(module, name, live.detach().clone())
                materialized = True
    return materialized


def _discriminating_operands(module: nn.Module):
    """Operands whose snapshot shape differs from the live transformed shape.

    These, and only these, can tell a checkpoint-shaped state dict apart from
    the model's own runtime ``state_dict()``. The CPU int4pack transform keeps
    ``scales`` at its checkpoint shape and the AVX512 transform keeps ``bias``
    at its checkpoint shape (only the dtype widens), so neither is decisive.
    """

    snapshot = getattr(module, "_fused_train_state", None) or {}
    for name, entry in snapshot.items():
        expected_shape = _entry_shape(entry)
        if expected_shape is None:
            expected = module._buffers.get(_entry_buffer_name(entry))
            if not isinstance(expected, torch.Tensor):
                continue
            expected_shape = tuple(expected.shape)
        live = getattr(module, name, None)
        if not isinstance(live, torch.Tensor):
            continue
        if expected_shape != tuple(live.shape):
            yield name, expected_shape


def classify_fused_state_layout(module: nn.Module, state_dict, prefix: str = "") -> str:
    """Classify the supplied local operands before changing the live tensors.

    A partial load can supply only one shape-changing operand. Every supplied
    discriminator must agree on its layout; missing operands are not evidence
    against that layout. Shared-shape operands alone use the live layout so a
    partial load cannot silently discard their new values. A load with no local
    operands leaves the transformed module untouched.
    """

    layout = None
    for name, expected_shape in _discriminating_operands(module):
        key = f"{prefix}{name}"
        if key not in state_dict:
            continue
        incoming = state_dict[key]
        live = getattr(module, name)
        if not isinstance(incoming, torch.Tensor):
            return "invalid"
        if tuple(incoming.shape) == expected_shape:
            candidate = "checkpoint"
        elif tuple(incoming.shape) == tuple(live.shape):
            candidate = "runtime"
        else:
            return "invalid"
        if layout is not None and layout != candidate:
            return "invalid"
        layout = candidate
    if layout is not None:
        return layout

    for name in (*module._parameters, *module._buffers):
        if name == QZEROS_FORMAT_BUFFER:
            # Layout metadata, not a weight: its presence says nothing about
            # whether the caller supplied checkpoint or runtime operands.
            continue
        if name not in module._non_persistent_buffers_set and f"{prefix}{name}" in state_dict:
            return "runtime"
    return "empty"


def state_dict_matches_checkpoint_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether supplied shape-changing operands match the checkpoint snapshot."""

    return classify_fused_state_layout(module, state_dict, prefix) == "checkpoint"


def state_dict_matches_runtime_layout(module: nn.Module, state_dict, prefix: str = "") -> bool:
    """Whether supplied operands must load into the live transformed layout."""

    return classify_fused_state_layout(module, state_dict, prefix) == "runtime"


__all__ = [
    "QZEROS_FORMAT_BUFFER",
    "RebuildCandidate",
    "RebuiltCheckpointTensor",
    "bind_runtime_qzeros",
    "_materialize_snapshot_entry",
    "capture_fused_train_state",
    "checkpoint_layout_tensor",
    "finalize_fused_train_state",
    "classify_fused_state_layout",
    "discard_fused_train_state",
    "fused_autograd_needs_dense",
    "materialize_fused_runtime_state",
    "refinalize_fused_state_after_apply",
    "reset_fused_runtime_state",
    "runtime_state_matches_live",
    "restore_fused_train_state",
    "state_dict_matches_checkpoint_layout",
    "state_dict_matches_runtime_layout",
]
