# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 candidate search must not discard every finite candidate.

The absolute FP32 SSE used to rank E4M3 block scales overflows to infinity for
large but finite activations.  With ``best_error`` also initialized to
infinity, no candidate ever replaced the all-zero default, so the quantizer
silently returned zeros.  Candidate ranking must stay finite and the search
must reject a block when no candidate has a usable scale.
"""

import math

import pytest
import torch

from gptqmodel.quantization.activation_floatx import (
    _nvfp4_mse_quantize_blocks,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)


pytestmark = [pytest.mark.cpu]

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual64 = actual.double()
    expected64 = expected.double()
    return (
        torch.linalg.vector_norm(actual64 - expected64)
        / torch.linalg.vector_norm(expected64)
    ).item()


def _e2m1_nearest(x: torch.Tensor) -> torch.Tensor:
    """Independent nearest-even rounding to the E2M1 magnitude grid."""
    magnitude = x.abs().clamp(max=_E2M1_VALUES[-1])
    grid = torch.tensor(_E2M1_VALUES, dtype=x.dtype, device=x.device)
    distance = (grid - magnitude[..., None]).abs()
    closest = distance == distance.min(dim=-1, keepdim=True).values
    even_code = closest & (torch.arange(len(_E2M1_VALUES), device=x.device) % 2 == 0)
    use_even = even_code.any(dim=-1, keepdim=True)
    chosen = torch.where(use_even, even_code, closest)
    values = grid[chosen.to(torch.int64).argmax(dim=-1)]
    return values.copysign(x)


def _e4m3_value(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _reference_relative_l2(x: torch.Tensor, global_scale: torch.Tensor) -> float:
    """Enumerate the finite positive E4M3 scales independently in FP64."""
    x64 = x.double().reshape(-1, 16)
    best = None
    for code in range(1, 127):
        scale = _e4m3_value(code) * float(global_scale)
        if not math.isfinite(scale) or scale <= 0:
            continue
        values = _e2m1_nearest(x64 / scale).double()
        error = ((values * scale - x64).square()).sum().item()
        if best is None or error < best[0]:
            best = (error, values * scale)
    return _relative_l2(best[1].reshape_as(x64), x64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("maximum", [1e17, 1e18, 1e19, 1e20, 1e30, 3e38])
def test_large_finite_activation_keeps_a_finite_candidate(dtype, maximum):
    x = torch.linspace(1.0, maximum, 16, dtype=torch.float64).to(dtype).reshape(1, 16)
    global_scale = nvfp4_global_scale(x.abs().amax())

    out = nvfp4_block_qdq(x, global_scale)

    assert out.dtype == dtype
    assert bool(torch.isfinite(out.float()).all())
    assert torch.count_nonzero(out).item() > 0
    relative = _relative_l2(out, x)
    reference = _reference_relative_l2(x.float(), global_scale)
    assert relative < 0.2
    # The production search is local, so allow it to land slightly above the
    # independent FP64 grid optimum without accepting the all-zero default.
    assert relative <= reference * 1.5


def test_search_rejects_when_no_candidate_has_a_usable_scale():
    blocks = torch.linspace(1.0, 1e20, 16).reshape(1, 1, 16)

    with pytest.raises(ValueError, match="no finite candidate"):
        _nvfp4_mse_quantize_blocks(blocks, torch.tensor(0.0))
