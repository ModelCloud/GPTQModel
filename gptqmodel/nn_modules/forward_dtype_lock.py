# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Forward dtype lock for modules that serve inference forwards.

Implements contracts/kernel/forward-dtype.md: the first eager forward records
the activation dtype; a later forward with another dtype, or a post-forward
``Module._apply`` that would change any parameter/buffer dtype, raises.
"""

from functools import wraps
from typing import Optional

import torch as t

from ..utils.torch import dynamo_is_compiling


FORWARD_DTYPE_CONTRACT = "contracts/kernel/forward-dtype.md"


def _forward_activation(args, kwargs) -> Optional[t.Tensor]:
    for value in args:
        if isinstance(value, t.Tensor):
            return value
    for key in ("x", "input", "hidden_states"):
        value = kwargs.get(key)
        if isinstance(value, t.Tensor):
            return value
    return None


def _check_forward_dtype(module: t.nn.Module, args, kwargs) -> None:
    """Enforce the forward dtype contract (contracts/kernel/forward-dtype.md).

    The first forward fixes the module's activation dtype. A later forward with
    a different dtype is a caller bug, so raise rather than adapt to it.
    """
    x = _forward_activation(args, kwargs)
    if x is None:
        return
    locked = getattr(module, "_forward_dtype_lock", None)
    if locked is None:
        # Never record from inside a dynamo trace: the write would make the
        # compiled graph guard on ``lock is None`` and force a recompile on
        # the next call. Eager calls record; traces only check an existing lock.
        if not dynamo_is_compiling():
            module._forward_dtype_lock = x.dtype
    elif x.dtype != locked:
        raise RuntimeError(
            f"{module.__class__.__name__}({getattr(module, 'name', '')}): forward received {x.dtype} "
            f"after the module was first called with {locked}. A module never switches "
            f"activation dtype during a run; see {FORWARD_DTYPE_CONTRACT}."
        )


def _lock_forward_dtype(forward):
    if getattr(forward, "_forward_dtype_locked", False):
        return forward

    @wraps(forward)
    def locked_forward(self, *args, **kwargs):
        _check_forward_dtype(self, args, kwargs)
        return forward(self, *args, **kwargs)

    locked_forward._forward_dtype_locked = True
    return locked_forward


def _check_apply_keeps_dtypes(module: t.nn.Module, fn) -> None:
    """Reject a post-forward ``_apply`` that would change a tensor dtype.

    After the first forward, weight and buffer dtypes are frozen
    (contracts/kernel/forward-dtype.md). ``fn`` is probed on an empty tensor
    of each dtype/device so the check runs before any tensor changes dtype.
    """
    locked = getattr(module, "_forward_dtype_lock", None)
    if locked is None:
        return
    probed = {}
    # Check the whole subtree up front so a rejected cast leaves every child
    # untouched, not only this module's own tensors.
    for kind, tensors in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in tensors:
            if not isinstance(tensor, t.Tensor):
                continue
            key = (tensor.dtype, tensor.device)
            if key not in probed:
                probe = fn(t.empty(0, dtype=tensor.dtype, device=tensor.device))
                probed[key] = probe.dtype if isinstance(probe, t.Tensor) else tensor.dtype
            if probed[key] != tensor.dtype:
                raise RuntimeError(
                    f"{module.__class__.__name__}({getattr(module, 'name', '')}): cannot change {kind} `{name}` "
                    f"from {tensor.dtype} to {probed[key]} after the module's first forward (input dtype {locked}). "
                    f"Weight dtypes are frozen after the first forward; see {FORWARD_DTYPE_CONTRACT}."
                )


def _lock_apply_dtypes(apply):
    if getattr(apply, "_forward_dtype_locked", False):
        return apply

    @wraps(apply)
    def locked_apply(self, fn, *args, **kwargs):
        _check_apply_keeps_dtypes(self, fn)
        return apply(self, fn, *args, **kwargs)

    locked_apply._forward_dtype_locked = True
    return locked_apply


# Entry points that rewrite a module's weights in place without ``_apply``.
_REWRITE_METHODS = ("hf_pack", "pack", "pack_block", "pack_gpu", "pack_original")


def _reject_after_forward(method, action: str):
    if getattr(method, "_forward_dtype_locked", False):
        return method

    @wraps(method)
    def guarded(self, *args, **kwargs):
        require_no_forward(self, action)
        return method(self, *args, **kwargs)

    guarded._forward_dtype_locked = True
    return guarded


class ForwardDtypeLockMixin:
    """Wrap every subclass ``forward`` and ``_apply`` with the forward dtype lock."""

    # The dtype of the first eager forward input; ``None`` until then.
    _forward_dtype_lock: Optional[t.dtype] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        forward = cls.__dict__.get("forward")
        if callable(forward):
            cls.forward = _lock_forward_dtype(forward)
        # Wrap the inherited ``nn.Module._apply`` too, so a module without its
        # own override still freezes its weight dtypes after the first forward.
        apply = getattr(cls, "_apply", None)
        if callable(apply):
            cls._apply = _lock_apply_dtypes(apply)
        for name in _REWRITE_METHODS:
            method = cls.__dict__.get(name)
            if callable(method):
                setattr(cls, name, _reject_after_forward(method, name.replace("_", " ")))


def require_no_forward(module: t.nn.Module, action: str) -> None:
    """Reject an in-place weight/dtype rewrite of a module that already ran a forward.

    Packing and dtype re-recording bypass ``Module._apply``, so they check
    the lock directly. Build a replacement module instead
    (contracts/kernel/forward-dtype.md).
    """
    locked = getattr(module, "_forward_dtype_lock", None)
    if locked is not None:
        raise RuntimeError(
            f"{module.__class__.__name__}({getattr(module, 'name', '')}): cannot {action} after the "
            f"module's first forward (input dtype {locked}). Weight dtypes are frozen after the first "
            f"forward; build a replacement module instead; see {FORWARD_DTYPE_CONTRACT}."
        )
