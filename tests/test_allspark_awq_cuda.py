# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ 8-bit validation for the AllSpark CUDA (Ampere+) kernel.

The HIP AWQ path lives in ``test_allspark_awq.py`` and is ROCm-gated. This file
covers the CUDA side of the same AWQ contract, which differs in exactly one
place worth testing:

* the CUDA kernel applies the zero point AFTER its bias-128 dequant, i.e. it
  computes ``((code - 128) - zero_buffer) * scale``, so the repack has to store
  ``zero_awq - 128``. Storing the raw AWQ zero instead shifts every weight by
  128 LSBs. ``test_awq_zero_point_convention_is_load_bearing`` pins that down
  with a negative control, because a plain accuracy assertion would still pass
  if both the repack and the oracle drifted the same way.

Oracle policy matches the rest of the repo: fp64 arithmetic from the original
codes/zeros/scales, weights rounded to the activation dtype.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.allspark import (
    allspark_gemm,
    allspark_repack_awq,
    allspark_supported,
    prewarm_allspark_extension,
)
from gptqmodel.utils.rocm import IS_ROCM


pytestmark = pytest.mark.skipif(
    IS_ROCM or not allspark_supported(),
    reason="AllSpark CUDA (Ampere+) GPU required",
)

DTYPES = [torch.float16, torch.bfloat16]


def _case(n, k, dtype, seed=0):
    """Real AWQ on-disk tensors from the repo's own packer."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 256, (k, n), generator=g, dtype=torch.int32).cuda()
    # Non-128 zeros on purpose: a neutral 128 zero would hide an off-by-128 bug.
    zeros = torch.randint(1, 256, (1, n), generator=g, dtype=torch.int32).cuda()
    scales = (
        (torch.rand(1, n, generator=g, dtype=torch.float32) * 0.01 + 0.001)
        .to(dtype)
        .cuda()
    )
    qweight, qzeros = pack_awq(codes, zeros, 8)
    return codes, zeros, scales, qweight, qzeros


def _oracle(a, codes, zeros, scales):
    exact = (codes.float() - zeros.float()) * scales.float()
    return a.double() @ exact.double()


def _check(a, codes, zeros, scales, qweight, qzeros, m, dtype):
    n, k = codes.shape[1], codes.shape[0]
    packed, pscales, pzeros = allspark_repack_awq(
        qweight, qzeros, scales, n, k, -1
    )
    out = allspark_gemm(a, packed, pscales, n, b_qzeros=pzeros)
    assert out.dtype == dtype
    assert out.shape == (m, n)

    ref64 = _oracle(a, codes, zeros, scales)
    # The gate is relative to max|ref| so it is independent of K's dynamic range.
    # fp16 rounds the 8-bit dequant product once per term; bf16 additionally
    # accumulates in bf16 (8-bit mantissa), so its floor is an order of magnitude
    # higher. Measured: fp16 <= 6.9e-4, bf16 <= 6.8e-3 at K=896.
    gate = 2e-3 if dtype == torch.float16 else 1.2e-2
    scale = ref64.abs().max().item()
    rel = (out.to(torch.float64) - ref64).abs().max().item() / scale
    assert rel <= gate, (
        f"m={m} k={k} dtype={dtype} rel={rel:.3e} exceeds {gate:.3e} "
        f"(max|ref|={scale:.3e})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n,k,m", [(256, 256, 1), (256, 256, 8), (4864, 896, 4)])
def test_awq_matches_fp64_oracle(dtype, n, k, m):
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    codes, zeros, scales, qweight, qzeros = _case(n, k, dtype, seed=n + k + m)
    a = torch.randn(m, k, device="cuda:0", dtype=dtype)
    _check(a, codes, zeros, scales, qweight, qzeros, m, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_zero_point_convention_is_load_bearing(dtype):
    """The stored zero buffer must be ``zero_awq - 128``, not the raw code.

    Feeds the raw zero through the same kernel and requires a visibly worse
    result, so a future "simplification" of the repack fails here instead of
    silently degrading every AWQ checkpoint.
    """
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    n, k, m = 256, 256, 4
    codes, zeros, scales, qweight, qzeros = _case(n, k, dtype, seed=99)
    a = torch.randn(m, k, device="cuda:0", dtype=dtype)

    packed, pscales, pzeros = allspark_repack_awq(qweight, qzeros, scales, n, k, -1)
    good = allspark_gemm(a, packed, pscales, n, b_qzeros=pzeros).float()

    # Negative control: undo the -128 adjustment.
    raw_zeros = (pzeros.float() + 128.0).to(dtype)
    bad = allspark_gemm(a, packed, pscales, n, b_qzeros=raw_zeros).float()

    ref = _oracle(a, codes, zeros, scales)
    ref_max = ref.abs().max().item()
    good_rel = (good.double() - ref).abs().max().item() / ref_max
    bad_rel = (bad.double() - ref).abs().max().item() / ref_max
    # Measured: good 2.6e-4..2.6e-3, bad 6.7e-1..7.3e-1.
    assert good_rel < 1e-2, f"correct convention should match, got rel={good_rel:.3e}"
    assert bad_rel > 0.05 and bad_rel > 10 * good_rel, (
        "raw-zero control did not degrade: the kernel may be ignoring b_qzeros "
        f"(good={good_rel:.3e}, bad={bad_rel:.3e})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_awq_rejects_grouped_zeros(dtype):
    """Only group_size=-1 is supported; anything else must raise, not mis-scale."""
    assert prewarm_allspark_extension(), "AllSpark CUDA extension failed to build"
    n, k = 256, 256
    _, _, scales, qweight, qzeros = _case(n, k, dtype, seed=5)
    with pytest.raises(ValueError, match="group_size=-1"):
        allspark_repack_awq(qweight, qzeros, scales, n, k, 128)
