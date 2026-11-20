# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Q4NX tensor-core GEMM selection and parity on Ada (sm_89).

The shared EXL3 GEMM tuning table predates the Q4NX codebook, so its CC_ADA
row is a copy of the 3-instruction sample.  That sample picks a 32-wide K tile
for 16-row batches and 96-128 cooperative blocks for narrow projections, which
is slower for Q4NX on a 128-SM part.  The dispatch now overrides both:

* ``size_m <= 16`` and ``size_n <= 8192`` run shape 2 (K tile 16, N tile 128);
* ``size_n <= 2048`` runs 64 cooperative blocks for K <= 4096 and 96 for wider
  K, because those shapes are bound by the split-K reduction, not by
  occupancy, and the wider K needs the extra blocks to keep 128 SMs fed.

This module pins the selection and checks the fused GEMM against the
reconstruct + HGEMM path the extension uses for the other codebooks.  Both
compute the same product in FP16, so they differ only by accumulation order.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.exllamav3.ext import exllamav3_ext, prewarm_exllamav3_extension
from tests.accelerator_gates import requires_nvidia_cuda


# Internal dispatch constants (see exl3_devctx.cuh and q4nx.cuh).
CC_ADA = 3
Q4NX_CODEBOOK = 3

# step bits -> (K, half_k) as the native dispatch sees them
STEP_BITS = {10: (2, True), 12: (3, False), 14: (3, True)}
# Packed tile geometry (EXL3_GEMM_TILESIZE_K/N in exl3_kernel_map.cuh).
TILESIZE_K = {1: 16, 2: 32, 3: 32, 4: 16, 5: 32, 6: 32, 7: 32}
TILESIZE_N = {1: 128, 2: 128, 3: 256, 4: 512, 5: 128, 6: 128, 7: 128}
BLOCKDIM = {1: 256, 2: 512, 3: 512, 4: 256, 5: 512, 6: 512, 7: 512}

# 1 / sqrt(42.5): the Q4NX normalisation the reconstruct path applies on output.
Q4NX_ALPHA = 1.0 / 42.5**0.5

# The two paths accumulate the same FP16 products in a different order.  The
# 180-shape sweep in tests/benchmark_exl3_q4nx_gemm_ada.py peaks at 8.0e-6
# against reference means of 0.04-0.09, so this bound still fails loudly on a
# wrong codebook, rotation or scale.
PARITY_ATOL = 2e-5

# Cooperative blocks for narrow (n <= 2048) projections: the split-K reduction
# is the bottleneck while K is small, and wider K needs more blocks to fill the
# 128 SMs.
NARROW_SMS = 64
NARROW_SMS_WIDE_K = 96
WIDE_K = 4096

requires_ada = pytest.mark.skipif(
    not requires_nvidia_cuda() or torch.cuda.get_device_capability() != (8, 9),
    reason="the Q4NX GEMM tuning is specific to Ada (sm_89)",
)

pytestmark = requires_ada


@pytest.fixture(scope="module")
def ops():
    """The compiled extension, or a precise skip when this build lacks it."""
    if not prewarm_exllamav3_extension():
        pytest.skip("the EXL3 torch.ops extension is not available in this build")
    return torch.ops.gptqmodel_exllamav3


def _probe(ops, m: int, k: int, n: int, step_bits: int):
    K, half_k = STEP_BITS[step_bits]
    kernel, shape_idx, num_sms, block_dim, q4nx_shape = ops.exl3_gemm_kernel_probe(
        CC_ADA, k, n, K, True, Q4NX_CODEBOOK, half_k, True, m
    )
    return kernel, shape_idx, num_sms, block_dim, q4nx_shape


def _row_shape(m: int) -> int:
    return 7 if m > 48 else 6 if m > 32 else 5 if m > 16 else 0


def _packed(k: int, n: int, step_bits: int, seed: int = 37):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, step_bits * 4), generator=generator, dtype=torch.int16
    ).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def _fused(ops, trellis, suh, svh, x, y, step_bits: int) -> int:
    return ops.bc_linear_exl3_run_configured(
        trellis,
        suh,
        svh,
        step_bits / 4.0,
        None,
        False,
        False,
        x,
        y,
        False,  # fp16_gemv
        0,  # int8_gemv
        True,  # fused_gemm
        True,  # q4nx
    )


def _reference(weight, suh, svh, x, y, xh) -> None:
    """The reconstruct branch of exl3_gemm: rotate, HGEMM, rescale."""
    exllamav3_ext.had_r_128(x, xh, suh, None, 1.0)
    exllamav3_ext.hgemm(xh, weight, y)
    exllamav3_ext.had_r_128(y, y, None, svh, Q4NX_ALPHA)


@pytest.mark.parametrize("step_bits", sorted(STEP_BITS))
@pytest.mark.parametrize("m", [1, 8, 16, 17, 32, 33, 48, 49, 64, 128, 256])
@pytest.mark.parametrize("k", [2048, 4096, 8192])
@pytest.mark.parametrize("n", [2048, 4096, 8192, 16384])
def test_q4nx_ada_selects_the_tuned_kernel(ops, step_bits, m, k, n):
    """The probe must report a Q4NX instance with the tuned geometry and grid."""
    kernel, shape_idx, num_sms, block_dim, q4nx_shape = _probe(ops, m, k, n, step_bits)

    assert kernel, "the dispatch returned no kernel"
    # The runtime only returns the shape index, which cannot tell a Q4NX
    # instance from a 3-instruction one with the same tile geometry.
    assert q4nx_shape == shape_idx, "the dispatch did not pick a Q4NX kernel instance"
    assert block_dim == BLOCKDIM[shape_idx]

    if m <= 16 and n <= 8192:
        # The shared sample's shape 3 (256-wide N tile) loses to shape 2.
        assert shape_idx == 2
    elif m <= 16:
        # Above 8192 columns the packed sample keeps its own choice.  For 3.0
        # bpw that is shape 3, which also proves the 8192 bound is applied.
        assert shape_idx in (1, 2, 3, 4)
        if step_bits == 12:
            assert shape_idx == 3
    else:
        # Multi-row tiles replace two or more slab passes and are unchanged.
        assert shape_idx == _row_shape(m)

    if n <= 2048:
        # Fewer blocks beat the 96-128 the shared sample asks for; the grid is
        # still clamped by the number of K x N tiles.
        max_slices = (k // TILESIZE_K[shape_idx]) * (n // TILESIZE_N[shape_idx])
        blocks = NARROW_SMS_WIDE_K if k > WIDE_K else NARROW_SMS
        assert num_sms == min(max_slices, blocks)


@pytest.mark.parametrize("step_bits", sorted(STEP_BITS))
@pytest.mark.parametrize("k", [2048, 4096, 8192])
@pytest.mark.parametrize("n", [2048, 4096, 8192, 16384])
def test_q4nx_fused_gemm_matches_reconstruct(ops, step_bits, k, n):
    """Fused GEMM == reconstruct + HGEMM up to FP16 accumulation order."""
    K, half_k = STEP_BITS[step_bits]
    trellis, suh, svh = _packed(k, n, step_bits)
    weight = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    ops.reconstruct_q4nx(weight, trellis, K, half_k)
    generator = torch.Generator().manual_seed(19)

    try:
        for m in (16, 32, 64, 128, 256):
            x = (torch.randn((m, k), generator=generator) * 0.1).half().cuda()
            xh = torch.empty_like(x)
            y = torch.empty((m, n), device="cuda", dtype=torch.float32)
            dispatch = _fused(ops, trellis, suh, svh, x, y, step_bits)
            _, shape_idx, _, _, q4nx_shape = _probe(ops, m, k, n, step_bits)
            assert dispatch == shape_idx == q4nx_shape, (
                f"dispatch {dispatch} disagrees with the probe {shape_idx} "
                f"for m={m} k={k} n={n} bits={step_bits}"
            )

            reference = torch.empty_like(y)
            _reference(weight, suh, svh, x, y=reference, xh=xh)
            assert reference.abs().mean() > 1e-3, "the reference product is degenerate"
            max_abs = (y - reference).abs().max().item()
            assert max_abs <= PARITY_ATOL, (
                f"fused GEMM differs from reconstruct by {max_abs:.3e} "
                f"for m={m} k={k} n={n} bits={step_bits}"
            )
    finally:
        del weight, trellis, suh, svh
        torch.cuda.empty_cache()
