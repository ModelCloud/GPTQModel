# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Load GPTQModel's vendored NF12 kernels without importing GPTQModel itself.

Nothing is loaded at import time. The source-backed build is owned by the
GPTQModel wheel and does not require an installation of the standalone package.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

import gptqmodel_nf12


_lock = threading.Lock()
_loaded = threading.Event()

_REQUIRED_OPS = (
    "encode",
    "decode",
    "linear",
    "embedding",
    "rowmajor_pack",
    "rowmajor_embedding",
    "encode_moe",
    "moe_linear",
    "_fused_moe",
)

_ANY_OPS = _REQUIRED_OPS + ("embedding_repeat4", "decode_fp32", "linear_staged_b1",
                            "pack_bf16t", "linear_bf16t", "moe_decode")


def _registered() -> bool:
    """Return whether a complete ``torch.ops.nf12`` implementation is already live."""
    namespace = torch.ops.nf12
    return all(callable(getattr(namespace, name, None)) for name in _REQUIRED_OPS)


def has_ops(*names: str) -> bool:
    """Return whether every named operator is registered in ``torch.ops.nf12``."""
    namespace = torch.ops.nf12
    return all(callable(getattr(namespace, name, None)) for name in names)


def _any_registered() -> bool:
    """Return whether *any* NF12 operator is already registered."""
    return any(callable(getattr(torch.ops.nf12, name, None)) for name in _ANY_OPS)


def require(*names: str) -> None:
    """Ensure the named operators are available, reusing a registered namespace.

    An older standalone NF12 library, or GPTQModel's legacy runtime, may own
    ``torch.ops.nf12`` with only a subset of the current operators. Loading a
    second library would fail registration, so reuse whatever is registered and
    only report the specific operators the caller needs.
    """
    if has_ops(*names):
        _loaded.set()
        return
    load()
    missing = [name for name in names if not callable(getattr(torch.ops.nf12, name, None))]
    if missing:
        raise RuntimeError(
            "gptqmodel_nf12: native operator(s) unavailable in the loaded NF12 library: "
            + ", ".join(missing)
        )


def _packaged() -> bool:
    """Return whether both native libraries ship with this checkout/wheel."""
    for directory in gptqmodel_nf12.__path__:
        base = Path(directory)
        if all((base / name).is_file() for name in ("libnf12.so", "libnf12_torch.so")):
            return True
    return False


def _native_available() -> bool:
    """Return whether the native extension can be used without a failed build attempt."""
    if _loaded.is_set() or _registered() or _packaged():
        return True
    from ._jit import cache_ready, source_build_available

    return cache_ready() or source_build_available()


def available() -> bool:
    """Return whether native NF12 kernels can run here.

    This is a capability probe, not a build: it reports whether the extension is
    already loaded/registered/packaged, a source build is cached for the visible
    GPUs, or the build tools needed for a first-use JIT build are present.
    Callers that can keep packed storage without native kernels should check
    this before selecting a native-only path.

    A namespace preloaded by an older standalone library satisfies this only
    when it exposes the complete current operator set. Use :func:`available_ops`
    to ask about a specific subset.
    """
    if torch.version.hip is not None:
        return False
    return _native_available()


def available_ops(*names: str) -> bool:
    """Return whether the named operators are registered or can be built here."""
    if torch.version.hip is not None:
        return False
    if has_ops(*names):
        return True
    if _any_registered():
        # A registered namespace owns the operator library; loading another one
        # would fail, so the answer is exactly what is already present.
        return False
    return _native_available()


def library_path(name: str = "libnf12.so") -> Path:
    """Find a packaged native library, or build and cache it for visible GPUs."""
    for directory in gptqmodel_nf12.__path__:
        candidate = Path(directory) / name
        if candidate.is_file():
            if name == "libnf12_torch.so" and not candidate.with_name("libnf12.so").is_file():
                continue
            return candidate
    if name in ("libnf12.so", "libnf12_torch.so"):
        from ._jit import library_directory

        return library_directory() / name
    raise ImportError(f"gptqmodel_nf12: native library {name} not found")


def load() -> None:
    """Idempotently loads the torch extension (``torch.ops.nf12``) on first use."""
    # The vendored native kernels are CUDA/CUTLASS implementations. PyTorch
    # exposes HIP tensors through the same ``cuda`` API, so check the build.
    if torch.version.hip is not None:
        raise RuntimeError("gptqmodel_nf12: native NF12 kernels support NVIDIA CUDA, not ROCm")
    if _loaded.is_set():
        return
    with _lock:
        if _loaded.is_set():
            return
        # GPTQModel's legacy runtime or an explicitly imported standalone NF12
        # may already own the torch namespace. A second TORCH_LIBRARY load
        # would fail registration, so reuse whatever implementation is already
        # registered; callers name the operators they need through require().
        if _registered() or _any_registered():
            _loaded.set()
            return
        # The Torch extension links the codec. Load it first so bare source
        # checkouts work without changing LD_LIBRARY_PATH.
        torch.ops.load_library(str(library_path("libnf12.so")))
        torch.ops.load_library(str(library_path("libnf12_torch.so")))
        _register_fakes()
        _loaded.set()


def _register_fakes() -> None:
    @torch.library.register_fake("nf12::encode")
    def _encode(w):
        ctx = torch.library.get_ctx()
        n, k = w.shape
        packed = w.new_empty((n, k * 3 // 2), dtype=torch.uint8)
        escapes = w.new_empty((ctx.new_dynamic_size(), 8), dtype=torch.int16)
        return packed, escapes, ctx.new_dynamic_size(min=0, max=120)

    @torch.library.register_fake("nf12::decode")
    def _decode(packed, escapes, exp_base):
        n, row_bytes = packed.shape
        return packed.new_empty((n, row_bytes * 2 // 3), dtype=torch.bfloat16)

    @torch.library.register_fake("nf12::decode_fp32")
    def _decode_fp32(packed, escapes, exp_base):
        n, row_bytes = packed.shape
        return packed.new_empty((n, row_bytes * 2 // 3), dtype=torch.float32)

    @torch.library.register_fake("nf12::sub_fp32")
    def _sub_fp32(packed, escapes, exp_base, wq):
        return torch.empty_like(wq)

    @torch.library.register_fake("nf12::linear")
    def _linear(x, packed, escapes, exp_base, bias=None, *, static_weights=False):
        return x.new_empty((*x.shape[:-1], packed.shape[0]))

    @torch.library.register_fake("nf12::linear_staged_b1")
    def _linear_staged_b1(x, packed, escapes, exp_base, bias=None):
        return x.new_empty((x.shape[0], packed.shape[0]))

    @torch.library.register_fake("nf12::pack_bf16t")
    def _pack_bf16t(w):
        n, k = w.shape
        return w.new_empty((n, 2 * k), dtype=torch.uint8)

    @torch.library.register_fake("nf12::linear_bf16t")
    def _linear_bf16t(x, w_bf16t, bias=None):
        return x.new_empty((*x.shape[:-1], w_bf16t.shape[0]))

    @torch.library.register_fake("nf12::embedding")
    def _embedding(ids, packed, escapes, exp_base):
        return packed.new_empty((*ids.shape, packed.shape[1] * 2 // 3), dtype=torch.bfloat16)

    @torch.library.register_fake("nf12::embedding_repeat4")
    def _embedding_repeat4(ids, packed, escapes, exp_base):
        return packed.new_empty((*ids.shape, 4, packed.shape[1] * 2 // 3), dtype=torch.bfloat16)

    @torch.library.register_fake("nf12::rowmajor_pack")
    def _rowmajor_pack(packed, escapes, exp_base):
        ctx = torch.library.get_ctx()
        n, row_bytes = packed.shape
        row_packed = packed.new_empty((n, row_bytes), dtype=torch.uint8)
        row_escapes = packed.new_empty((ctx.new_dynamic_size(), 4), dtype=torch.int16)
        return (row_packed, row_escapes, ctx.new_dynamic_size(min=0),
                ctx.new_dynamic_size(min=0))

    @torch.library.register_fake("nf12::rowmajor_embedding")
    def _rowmajor_embedding(ids, row_packed, row_escapes, exp_base):
        return row_packed.new_empty(
            (*ids.shape, row_packed.shape[1] * 2 // 3), dtype=torch.bfloat16
        )

    @torch.library.register_fake("nf12::encode_moe")
    def _encode_moe(w):
        ctx = torch.library.get_ctx()
        e, n, k = w.shape
        packed = w.new_empty((e, n, k * 3 // 2), dtype=torch.uint8)
        escapes = w.new_empty((ctx.new_dynamic_size(), 8), dtype=torch.int16)
        return packed, escapes, w.new_empty((e + 1,), dtype=torch.int64), w.new_empty((e,), dtype=torch.uint8)

    @torch.library.register_fake("nf12::moe_align")
    def _moe_align(topk_ids, num_experts):
        return (topk_ids.new_empty((topk_ids.numel(),), dtype=torch.int32),
                topk_ids.new_empty((num_experts + 1,), dtype=torch.int64))

    @torch.library.register_fake("nf12::moe_decode")
    def _moe_decode(packed, escapes, escape_offsets, exp_base):
        e, n, row_bytes = packed.shape
        return packed.new_empty((e, n, row_bytes * 2 // 3), dtype=torch.bfloat16)

    def _moe_rows(x, gather, scatter):
        return gather.shape[0] if gather is not None else scatter.shape[0] if scatter is not None else x.shape[0]

    @torch.library.register_fake("nf12::moe_linear")
    def _moe_linear(x, packed, escapes, escape_offsets, exp_base, expert_offsets, gather=None, gather_div=1,
                    scatter=None, row_scale=None, zero_fill=False):
        return x.new_empty((_moe_rows(x, gather, scatter), packed.shape[1]))

    @torch.library.register_fake("nf12::moe_linear_bf16t")
    def _moe_linear_bf16t(x, w_bf16t, expert_offsets, gather=None, gather_div=1, scatter=None, row_scale=None,
                          zero_fill=False):
        return x.new_empty((_moe_rows(x, gather, scatter), w_bf16t.shape[1]))

    @torch.library.register_fake("nf12::moe_combine")
    def _moe_combine(y, topk):
        return y.new_empty((y.shape[0] // topk, y.shape[1]))

    @torch.library.register_fake("nf12::moe_silu_mul")
    def _moe_silu_mul(h):
        return h.new_empty((h.shape[0], h.shape[1] // 2))

    @torch.library.register_fake("nf12::_fused_moe")
    def _fused_moe(x, packed13, escapes13, escape_offsets13, exp_base13,
                   packed2, escapes2, escape_offsets2, exp_base2, topk_weights, topk_ids, zero_fill=False):
        return x.new_empty((x.shape[0], packed2.shape[1]))
