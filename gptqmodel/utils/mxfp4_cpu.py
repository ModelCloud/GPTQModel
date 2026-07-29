# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the MXFP4 AVX-512 CPU kernel: JIT loader and MXFP4 quant/dequant."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load


# MXFP4 E2M1 value table (index = 4-bit nibble).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _quantization_thresholds() -> torch.Tensor:
    """Thresholds between FP4 positive magnitude bins."""
    return torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def quantize_mxfp4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense (N, K) float32 weight to MXFP4.

    Returns:
        qweight: (N, K//2) uint8 with two E2M1 nibbles per byte.
        scales: (N, K//32) uint8 E8M0 scales.
    """
    weight = weight.to(torch.float32)
    N, K = weight.shape
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {K}")

    w_blocks = weight.view(N, K // 32, 32)
    max_abs = w_blocks.abs().amax(dim=-1, keepdim=True)

    needs = max_abs / 6.0
    needs = torch.where(needs <= 0, torch.tensor(1.0, dtype=torch.float32), needs)
    log2 = torch.log2(needs)
    exp = torch.ceil(log2).clamp(-127, 127)
    scale_f = torch.exp2(exp)
    scale_bits = (exp + 127).to(torch.int32).clamp(0, 254).to(torch.uint8)

    scaled = w_blocks / scale_f

    pos_values = FP4_TABLE[:8].to(weight.device)
    thresholds = _quantization_thresholds().to(weight.device)

    abs_scaled = scaled.abs()
    edges = torch.cat([torch.tensor([0.0], device=weight.device), thresholds])
    idx = torch.searchsorted(edges, abs_scaled, right=False)
    idx = idx.clamp(0, 7)
    quantized_abs = pos_values[idx]
    quantized = torch.where(scaled >= 0, quantized_abs, -quantized_abs)

    dist = (quantized.unsqueeze(-1) - FP4_TABLE.to(weight.device)).abs()
    nibbles = dist.argmin(dim=-1).to(torch.uint8)

    nibbles = nibbles.view(N, K // 32, 32)
    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    bytes_ = (even & 0x0F) | ((odd & 0x0F) << 4)
    qweight = bytes_.view(N, K // 2)

    scales = scale_bits.squeeze(-1)
    return qweight, scales


def dequantize_mxfp4(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 weights to a dense (N, K) float32 matrix."""
    N, K_half = qweight.shape
    K = K_half * 2
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {K}")

    low = qweight & 0x0F
    high = (qweight >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).view(N, K)

    scale_f = torch.exp2(scales.to(torch.float32) - 127.0)
    scale_f = scale_f.unsqueeze(-1).repeat(1, 1, 32).view(N, K)

    values = FP4_TABLE.to(qweight.device)[nibbles.to(torch.int64)] * scale_f
    return values


def _ensure_ninja_on_path() -> None:
    """Make the ninja binary discoverable when only the PyPI package is installed."""
    if shutil.which("ninja"):
        return
    try:
        import ninja
        bin_dir = Path(ninja.BIN_DIR)
        ninja_bin = bin_dir / "ninja"
        if not ninja_bin.exists():
            return
    except (ImportError, AttributeError, OSError):
        return
    path = os.environ.get("PATH", "")
    bin_dir_str = str(bin_dir)
    if bin_dir_str in path.split(os.pathsep):
        return
    os.environ["PATH"] = f"{bin_dir_str}{os.pathsep}{path}"


_FP16_PROBE = """
#include <immintrin.h>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512fp16")))
__m512h probe(__m512h a, __m512h b, __m512h c) { return _mm512_fmadd_ph(a, b, c); }
int main() { return 0; }
"""


def _compiler_has_avx512fp16(compiler: str) -> bool:
    """Return True when `compiler` understands the avx512fp16 target attribute."""
    if shutil.which(compiler) is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cpp"
        src.write_text(_FP16_PROBE)
        proc = subprocess.run(
            [compiler, "-std=c++17", "-c", str(src), "-o", str(Path(tmp) / "probe.o")],
            capture_output=True,
            check=False,
        )
    return proc.returncode == 0


def _select_compiler() -> None:
    """Point torch's JIT at a compiler new enough for the AVX512-FP16 intrinsics."""
    current = os.environ.get("CXX", "c++")
    if _compiler_has_avx512fp16(current):
        return
    for candidate in ("g++-14", "g++-13", "g++-12", "clang++-16", "clang++-15", "clang++-14"):
        if _compiler_has_avx512fp16(candidate):
            os.environ["CXX"] = candidate
            return


_EXTENSION: Optional[object] = None


def load_mxfp4_cpu_kernel() -> object:
    """Build/load the experimental C++ MXFP4 kernel via torch.utils.cpp_extension (Ninja)."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    _ensure_ninja_on_path()
    _select_compiler()

    src = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "mxfp4_cpu_kernel.cpp"
    if not src.exists():
        raise FileNotFoundError(src)

    extra_cflags = [
        "-O3",
        "-std=c++17",
        "-fopenmp",
    ]
    extra_cflags += os.environ.get("GPTQMODEL_MXFP4_EXTRA_CFLAGS", "").split()
    extra_ldflags = ["-fopenmp"]

    _EXTENSION = load(
        name="mxfp4_cpu_kernel",
        sources=[str(src)],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        is_python_module=True,
        verbose=os.environ.get("GPTQMODEL_MXFP4_VERBOSE", "0") == "1",
    )
    return _EXTENSION
