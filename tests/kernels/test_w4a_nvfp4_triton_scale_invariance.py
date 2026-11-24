# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 activation packing must rank block scales like the replay reference.

The replay reference (``_nvfp4_mse_quantize_blocks``) compares E4M3 candidates
and solves the least-squares scale in per-block power-of-two coordinates.  The
Triton packer used absolute FP32 units, so a block scaled by a large or small
power of two tied at ``inf`` or ``0`` and selected a different scale than
replay.  These tests sweep an exact ``2**k`` rescale and compare the complete
packed bytes and swizzled scales with the replay reference, and the selected
scale and packed codes with an independent FP64 oracle.

Coverage and limits:

* recipes: ``nvidia``, ``four_six``, ``least_squares``, ``least_squares_grid``;
  the headroom names map to the same two kernels.
* dtypes: BF16 and FP16, each on its own input tensor.  No kernel module is
  reused across dtypes, so the forward dtype contract is not exercised here.
* hardware: CUDA with the Triton E4M3 PTX conversion.  This does not exercise
  the NVFP4 tensor-core forward, which needs its own device.
* FTZ: normalized operands are O(1); subnormal inputs and global scales are
  pre-scaled exactly before ``ldexp``.  The fixtures include BF16 subnormal
  blocks and subnormal FP32 global scales.
* rounding: exact equality is required, so no midpoint tie is adjudicated.
  E4M3 midpoint coverage lives in
  ``test_w4a_nvfp4_gb10.py::test_nvfp4_block_scale_rounding_holds_at_e4m3_midpoint_neighbours``.
"""

from __future__ import annotations

import math

import pytest
import torch


pytest.importorskip("triton")

from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import _swizzle_scales, nvfp4_input
from gptqmodel.nn_modules.qlinear.w4a_nvfp4_triton import (
    nvfp4_decode,
    nvfp4_pack_and_swizzle,
)
from gptqmodel.quantization.activation_floatx import nvfp4_block_qdq
from tests.w4a_hardware_marks import FP8_HARDWARE


pytestmark = [FP8_HARDWARE]

_BF16_EXPONENTS = (0, 60, 80, 100, 124, 126, -60, -80, -100, -120)
_FP16_EXPONENTS = (0, 8, 12, 15, -14, -20, -24)

# A fixed block whose Four-Over-Six seeds are E4M3 208 (maximum 4) and 144
# (maximum 6).  Least-squares refinement recovers 144; the seed-only recipe
# keeps 208.
_FOUR_OVER_SIX_BLOCK = (
    0.4472350478172302, 0.2393692433834076, -0.028780341148376465,
    0.43915855884552, -0.6220099329948425, -0.3339952528476715,
    -0.0813106968998909, 0.802413821220398, 0.5239754319190979,
    -0.25585582852363586, -0.0012009433703497052, 0.22150030732154846,
    -0.8143987655639648, 0.027795936912298203, 0.6136912107467651,
    -0.5106378793716431,
)

# A block whose better Four-Over-Six seed is the maximum-6 one (192), so the
# seed comparison itself is exercised, not just the refinement.
_SIX_BLOCK = (
    -0.30484700202941895, 1.0453333854675293, -0.2785641849040985,
    0.7178201079368591, -0.7942410707473755, -1.1229435205459595,
    -0.8032764196395874, -1.0096871852874756, 0.6302139759063721,
    -0.28723299503326416, 0.4071083068847656, 0.7822999358177185,
    0.3319821059703827, 1.1566801071166992, 0.8740622997283936,
    0.381226122379303,
)

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_value(code: int) -> float:
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def _e2m1_code_index(values: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code, ties to even, using numeric distances."""
    magnitude = values.abs().clamp(max=6.0)
    grid = torch.tensor(_E2M1_VALUES, dtype=values.dtype, device=values.device)
    distance = (grid - magnitude[..., None]).abs()
    closest = distance == distance.min(dim=-1, keepdim=True).values
    even = closest & (torch.arange(len(grid), device=values.device) % 2 == 0)
    chosen = torch.where(even.any(dim=-1, keepdim=True), even, closest)
    return chosen.to(torch.int64).argmax(dim=-1)


def _e2m1_nearest_values(values: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1_VALUES, dtype=values.dtype, device=values.device)
    return grid[_e2m1_code_index(values)].copysign(values)


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    code = _e2m1_code_index(values).to(torch.uint8)
    return code | ((values < 0).to(torch.uint8) << 3)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        raise ValueError("E2M1 codes need an even last dimension.")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _best_global_scale(block: torch.Tensor, global_scale: float) -> float:
    """Enumerate every positive finite E4M3 scale in FP64 and return the best."""
    block = block.double()
    best = None
    for code in range(1, 127):
        local = _e4m3_value(code)
        scale = local * global_scale
        if not math.isfinite(scale) or scale <= 0:
            continue
        values = _e2m1_nearest_values(block / scale)
        error = ((values * scale - block).square().sum()).item()
        if best is None or error < best[0]:
            best = (error, local)
    assert best is not None
    return best[1]


def _four_six_scale(block: torch.Tensor, global_scale: float) -> float:
    """Evaluate only the two hardware seed scales, as the legacy recipe does."""
    block = block.double()
    maximum = block.abs().max().float()
    best = None
    for bound in (4.0, 6.0):
        inverse = (maximum / (bound * global_scale)).clamp(2.0**-9, 448.0)
        local = float(inverse.to(torch.float8_e4m3fn).float())
        scale = local * global_scale
        values = _e2m1_nearest_values(block / scale)
        error = ((values * scale - block).square().sum()).item()
        if best is None or error < best[0]:
            best = (error, local)
    return best[1]


def _expected_local(block: torch.Tensor, global_scale: float, recipe: str) -> float:
    if recipe == "four_six":
        return _four_six_scale(block, global_scale)
    return _best_global_scale(block, global_scale)


def _unswizzle_scales(swizzled: torch.Tensor, rows: int = 1, cols: int = 8) -> torch.Tensor:
    """Invert the 32x4x4 scale swizzle for a single-row fixture."""
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    index = torch.arange(
        padded_rows * padded_cols, device=swizzled.device, dtype=torch.long
    ).reshape(padded_rows, padded_cols)
    mapping = (
        index.view(padded_rows // 128, 128, padded_cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )
    out = torch.zeros(rows * cols, dtype=swizzled.dtype, device=swizzled.device)
    real = mapping < rows * cols
    out[mapping[real]] = swizzled.reshape(-1)[: mapping.numel()][real]
    return out


def _fixture(block: tuple[float, ...], dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.tensor(block, dtype=dtype, device="cuda")
        .reshape(1, 16)
        .repeat(1, 8)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "recipe", ["nvidia", "four_six", "least_squares", "least_squares_grid"]
)
def test_packed_payload_matches_replay_across_power_of_two_rescale(dtype, recipe):
    block = _SIX_BLOCK if recipe == "four_six" else _FOUR_OVER_SIX_BLOCK
    base = _fixture(block, dtype)
    exponents = _BF16_EXPONENTS if dtype == torch.bfloat16 else _FP16_EXPONENTS

    for exponent in exponents:
        scale = 2.0**exponent
        x = base * scale
        global_scale = torch.tensor(
            0.001 * scale, dtype=torch.float32, device="cuda"
        )
        assert bool(torch.isfinite(x).all())
        assert bool(torch.isfinite(global_scale)) and float(global_scale) > 0

        packed, swizzled = nvfp4_pack_and_swizzle(x, global_scale, recipe=recipe)
        reference_packed, reference_local = nvfp4_input(
            x, global_scale, recipe=recipe
        )
        torch.cuda.synchronize()

        assert torch.equal(
            packed.view(torch.uint8), reference_packed.view(torch.uint8)
        ), f"{recipe} {dtype} 2**{exponent}: packed codes differ from replay"
        expected_scales = _swizzle_scales(reference_local)
        assert torch.equal(
            swizzled.view(torch.uint8).reshape(-1),
            expected_scales.view(torch.uint8).reshape(-1),
        ), f"{recipe} {dtype} 2**{exponent}: swizzled scales differ from replay"

        decoded = nvfp4_decode(
            packed, swizzled, global_scale, x.shape[1], dtype
        )
        reference = nvfp4_block_qdq(x, global_scale, recipe)
        assert torch.equal(
            decoded, reference
        ), f"{recipe} {dtype} 2**{exponent}: decoded payload differs from replay"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("recipe", ["four_six", "least_squares", "least_squares_grid"])
@pytest.mark.parametrize(
    "block",
    [_FOUR_OVER_SIX_BLOCK, _SIX_BLOCK],
    ids=["four_over_six_block", "six_block"],
)
def test_pack_selects_the_independent_oracle_scale(dtype, recipe, block):
    base = _fixture(block, dtype)
    exponents = (0, 124, -120) if dtype == torch.bfloat16 else (0, 15, -24)

    for exponent in exponents:
        scale = 2.0**exponent
        x = base * scale
        global_scale_value = 0.001 * scale
        global_scale = torch.tensor(
            global_scale_value, dtype=torch.float32, device="cuda"
        )
        packed, swizzled = nvfp4_pack_and_swizzle(x, global_scale, recipe=recipe)

        local = (
            _unswizzle_scales(swizzled.view(torch.uint8).reshape(-1))
            .view(torch.float8_e4m3fn)
            .float()
        )
        # Use the tensor the kernel actually consumed: an FP16 subnormal
        # product can round differently from the exact FP32 rescale.
        oracle_block = x[0].float()
        expected_local = _expected_local(oracle_block, global_scale_value, recipe)
        assert torch.equal(
            local, torch.full_like(local, expected_local)
        ), f"{recipe} {dtype} 2**{exponent}: selected {local[0].item()}, oracle {expected_local}"

        normalized = oracle_block.reshape(1, 8, 16) / (
            expected_local * global_scale_value
        )
        expected_packed = _pack_codes(
            _e2m1_codes(normalized).reshape(1, 128)
        )
        assert torch.equal(
            packed.view(torch.uint8), expected_packed
        ), f"{recipe} {dtype} 2**{exponent}: packed codes differ from the oracle"
