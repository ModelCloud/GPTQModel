# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Optional PyTorch CPU helpers for ttbfp.

These are thin zero-copy wrappers around the NumPy API; they are convenient
for testing models that use TT block-float weights without writing ctypes
plumbing.  CPU tensors only (a CUDA tensor would need the optional
libttbfp_cuda.so path).

    from ttbfp import torch_ops
    exps, data = torch_ops.pack(w, "ttbf8")
    y = torch_ops.unpack(exps, data, "ttbf8", cols=w.shape[1])
    c = torch_ops.matmul(a, exps, data, "ttbf8", cols=w.shape[1])
"""

from __future__ import annotations

import torch

import ttbfp


def _cpu_f32(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if x.device.type != "cpu":
        raise ValueError("ttbfp.torch_ops is CPU-only; use the CUDA API for GPU")
    return x.detach().to(torch.float32).contiguous()


def pack(x: torch.Tensor, fmt=ttbfp.TTBF8):
    """Quantize a [rows, cols] fp32 CPU tensor -> (exps, data) uint8 tensors."""
    x = _cpu_f32(x)
    if x.dim() != 2:
        raise ValueError("pack expects a 2-D tensor")
    e, d = ttbfp.pack(x.numpy(), fmt)
    return torch.from_numpy(e), torch.from_numpy(d)


def unpack(exps: torch.Tensor, data: torch.Tensor, fmt=ttbfp.TTBF8,
           cols: int | None = None, ftz: bool = True) -> torch.Tensor:
    e = exps.detach().cpu().contiguous().numpy()
    d = data.detach().cpu().contiguous().numpy()
    return torch.from_numpy(ttbfp.unpack(e, d, fmt, cols=cols, ftz=ftz))


def matmul(a: torch.Tensor, exps: torch.Tensor, data: torch.Tensor,
           fmt=ttbfp.TTBF8, cols: int | None = None,
           ftz: bool = True) -> torch.Tensor:
    """C = A @ dequant(B) for CPU tensors; B stays packed."""
    a = _cpu_f32(a)
    e = exps.detach().cpu().contiguous().numpy()
    d = data.detach().cpu().contiguous().numpy()
    return torch.from_numpy(ttbfp.matmul(a.numpy(), e, d, fmt, cols=cols, ftz=ftz))


def matmul_bf16(a_bf16: torch.Tensor, exps: torch.Tensor, data: torch.Tensor,
                fmt=ttbfp.TTBF8, cols: int | None = None,
                ftz: bool = True) -> torch.Tensor:
    """Like matmul(), with A as bf16 bit patterns (torch.uint16)."""
    if a_bf16.device.type != "cpu":
        raise ValueError("ttbfp.torch_ops is CPU-only")
    a = a_bf16.detach().cpu().contiguous().numpy()
    e = exps.detach().cpu().contiguous().numpy()
    d = data.detach().cpu().contiguous().numpy()
    return torch.from_numpy(ttbfp.matmul_bf16(a, e, d, fmt, cols=cols, ftz=ftz))


def pack_transposed(x: torch.Tensor, fmt=ttbfp.TTBF8):
    """Quantize and return the block-major layout (best for repeated matmul)."""
    x = _cpu_f32(x)
    if x.dim() != 2:
        raise ValueError("pack_transposed expects a 2-D tensor")
    e, d = ttbfp.pack_transposed(x.numpy(), fmt)
    return torch.from_numpy(e), torch.from_numpy(d)


def matmul_prepacked(a: torch.Tensor, exps_t: torch.Tensor,
                     data_t: torch.Tensor, fmt=ttbfp.TTBF8,
                     cols: int | None = None, ftz: bool = True) -> torch.Tensor:
    """C = A @ dequant(B) with B in the block-major layout."""
    a = _cpu_f32(a)
    et = exps_t.detach().cpu().contiguous().numpy()
    dt = data_t.detach().cpu().contiguous().numpy()
    return torch.from_numpy(
        ttbfp.matmul_prepacked(a.numpy(), et, dt, fmt, cols=cols, ftz=ftz))


def prepack_s8(x: torch.Tensor, fmt=ttbfp.TTBF8):
    """Quantize into the int8 kernel-native block-major layout."""
    x = _cpu_f32(x)
    if x.dim() != 2:
        raise ValueError("prepack_s8 expects a 2-D tensor")
    e, s8 = ttbfp.prepack_s8(x.numpy(), fmt)
    return torch.from_numpy(e), torch.from_numpy(s8)


def matmul_prepacked_s8(a: torch.Tensor, exps_t: torch.Tensor,
                        s8: torch.Tensor, fmt=ttbfp.TTBF8,
                        cols: int | None = None, ftz: bool = True) -> torch.Tensor:
    """C = A @ dequant(B) with B in the int8 kernel-native layout."""
    a = _cpu_f32(a)
    et = exps_t.detach().cpu().contiguous().numpy()
    st = s8.detach().cpu().contiguous().numpy()
    return torch.from_numpy(
        ttbfp.matmul_prepacked_s8(a.numpy(), et, st, fmt, cols=cols, ftz=ftz))
