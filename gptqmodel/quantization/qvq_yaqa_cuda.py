# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict FP32 projection tiles for Hopper Sketch-B collection."""

import threading
from functools import cache

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


_PROJECT_WARM_KEYS: set[tuple[object, ...]] = set()
_PROJECT_WARM_KEYS_LOCK = threading.Lock()


def _require_project_warm(key: tuple[object, ...]) -> None:
    """Reject first Triton compilation from inside a CUDA Graph capture."""

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        with _PROJECT_WARM_KEYS_LOCK:
            warm = key in _PROJECT_WARM_KEYS
        if not warm:
            raise RuntimeError(
                "QVQ YAQA Triton projection must be warmed before CUDA Graph capture"
            )


def _mark_project_warm(key: tuple[object, ...]) -> None:
    with _PROJECT_WARM_KEYS_LOCK:
        _PROJECT_WARM_KEYS.add(key)


if triton is not None:

    @triton.jit
    def _project_kernel(
        activation, projection, output,
        K: tl.constexpr, A0: tl.constexpr, A1: tl.constexpr,
        P0: tl.constexpr, P1: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
    ):
        # Each output accumulates increasing k with FP32 FMA, without split-K,
        # TF32, or reassociation across partial sums. Only output tiling changes.
        m = tl.program_id(0) * BM + tl.arange(0, BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        batch = tl.program_id(2)
        offsets = tl.arange(0, 64)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(K // 64):
            k = block * 64 + offsets
            a = tl.load(activation + batch * A0 + m[:, None] * A1 + k[None, :])
            p = tl.load(projection + batch * P0 + k[:, None] * P1 + n[None, :])
            acc = tl.dot(a, p, acc, input_precision="ieee")
        tl.store(output + batch * 64 * 256 + m[:, None] * 256 + n[None, :], acc)


@cache
def _is_hopper(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (9, 0)


def project(activation: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """Project tested long-channel, 64-token batches; retain bmm elsewhere.

    The collector uses IEEE FP32 statistics. Restrict this specialization to the
    measured geometry and unit channel/rank strides, including sliced Gaussian
    projection batches. Autograd and non-Hopper callers keep the torch operator.
    """
    if (
        triton is None
        or activation.device.type != "cuda"
        or activation.device != projection.device
        or activation.dtype != torch.float32
        or projection.dtype != torch.float32
        or activation.ndim != 3
        or projection.ndim != 3
        or activation.requires_grad
        or projection.requires_grad
        or getattr(torch.backends.cuda.matmul, "fp32_precision", None) != "ieee"
    ):
        return torch.bmm(activation, projection)
    batch, tokens, channels = activation.shape
    if (
        not 2 <= batch <= 16
        or tokens != 64
        or channels not in (5120, 17408)
        or projection.shape != (batch, channels, 256)
        or activation.stride(2) != 1
        or projection.stride(2) != 1
        or not _is_hopper(activation.device)
    ):
        return torch.bmm(activation, projection)
    bm, bn = (16, 16) if batch <= 4 else (32, 64)
    key = (
        activation.device.type,
        activation.device.index,
        "project",
        batch,
        tokens,
        channels,
        bm,
        bn,
        activation.stride(0),
        activation.stride(1),
        projection.stride(0),
        projection.stride(1),
    )
    _require_project_warm(key)
    output = torch.empty((batch, tokens, 256), dtype=torch.float32, device=activation.device)
    _project_kernel[(tokens // bm, 256 // bn, batch)](
        activation, projection, output, channels,
        activation.stride(0), activation.stride(1), projection.stride(0), projection.stride(1),
        bm, bn, num_warps=4,
    )
    _mark_project_warm(key)
    return output
