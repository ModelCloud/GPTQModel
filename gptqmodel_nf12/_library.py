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


def library_path(name: str = "libnf12_ad.so") -> Path:
    """Find a packaged native library, or build and cache it for visible GPUs."""
    for directory in gptqmodel_nf12.__path__:
        candidate = Path(directory) / name
        if candidate.is_file():
            if name == "libnf12_ad_torch.so" and not candidate.with_name("libnf12_ad.so").is_file():
                continue
            return candidate
    if name in ("libnf12_ad.so", "libnf12_ad_torch.so"):
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
        # would fail, so reuse a complete registered implementation.
        namespace = torch.ops.nf12
        required = ("encode", "decode", "linear", "embedding", "encode_moe",
                    "moe_linear", "_fused_moe")
        if all(callable(getattr(namespace, name, None)) for name in required):
            _loaded.set()
            return
        # The Torch extension links the codec. Load it first so bare source
        # checkouts work without changing LD_LIBRARY_PATH.
        torch.ops.load_library(str(library_path("libnf12_ad.so")))
        torch.ops.load_library(str(library_path("libnf12_ad_torch.so")))
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

    @torch.library.register_fake("nf12::linear")
    def _linear(x, packed, escapes, exp_base, bias=None):
        return x.new_empty((*x.shape[:-1], packed.shape[0]))

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
