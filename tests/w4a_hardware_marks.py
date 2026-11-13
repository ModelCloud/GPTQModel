# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hardware gates shared by the W4A test matrix.

Native E4M3 tensor-core products start with Ada Lovelace (SM 8.9), so the FP8
carrier runs on any SM 8.9+ part; Hopper and Blackwell qualify as well. The
block-scaled FP4 carrier is validated on GB10 / SM 12.1 only in this release.

The kernels themselves are NVIDIA-only: the FP8 GEMM lowers to
``ttg.nvidia_mma`` and every E4M3 rounding step is PTX. HIP reuses the CUDA
device namespace, so ``torch.cuda.is_available()`` and
``torch.cuda.get_device_capability`` both report plausible values on ROCm
(gfx1201 reports ``(12, 0)``) even though Triton cannot compile the kernels
there. The gates below therefore also require a non-ROCm platform.

Tests that parametrize several carriers attach the mark to the parameter
(``pytest.param("w4a_nvfp4", marks=NVFP4_HARDWARE)``) instead of decorating the
function, so a single file can exercise the FP8 lane on Ada while its NVFP4
cases stay skipped.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import normalize_device
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear


def w4a_kernels_available() -> bool:
    """Whether this host can run any W4A Triton kernel.

    The kernels are CUDA-only by construction (``ttg.nvidia_mma`` GEMM operands
    and PTX E4M3 rounding), so the platform has to be checked explicitly:
    ``torch.cuda.is_available()`` is true on ROCm because HIP reuses the CUDA
    device namespace, and ``get_device_capability`` reports a synthetic SM
    version there. Use this instead of the torch device namespace when gating a
    test that launches a W4A kernel.

    The answer comes from the kernel's own ``SUPPORTS_DEVICES`` declaration
    compared against the resolved default family, so enabling ROCm later only
    needs the declaration to change.
    """
    return normalize_device(0) in W4AFP8Linear.SUPPORTS_DEVICES


def _capability() -> tuple[int, int] | None:
    if not w4a_kernels_available():
        return None
    return torch.cuda.get_device_capability(0)


def _has_native_fp8_gemm() -> bool:
    """Whether this Triton build exposes the Gluon ``mma_v2`` FP8 GEMM.

    ``mma_v2`` first appears in Triton 3.6.0, and not every build provides it.
    Probe the symbol instead of comparing version strings so a backport or a
    nightly build that carries the API is accepted.
    """
    try:
        from triton.experimental.gluon.language.nvidia.ampere import mma_v2  # noqa: F401
    except ImportError:
        return False
    return True


FP8_HARDWARE = pytest.mark.skipif(
    _capability() is None or _capability() < (8, 9),
    reason="FP8 tensor cores (SM 8.9+ / Ada or newer) required",
)
NVFP4_HARDWARE = pytest.mark.skipif(
    _capability() != (12, 1),
    reason="GB10 / SM 12.1 required for NVFP4",
)
# Every W4A carrier is packed and multiplied by the Triton kernels, so a build
# without the Gluon `mma_v2` FP8 GEMM cannot run any of them. The Gluon
# ``nvidia.ampere`` module also imports on a HIP build, where it is unusable, so
# the platform gate is required here as well.
GLUON_FP8 = pytest.mark.skipif(
    not w4a_kernels_available() or not _has_native_fp8_gemm(),
    reason="Triton with Gluon `mma_v2` FP8 GEMM (>= 3.6.0) on an NVIDIA CUDA device required",
)
