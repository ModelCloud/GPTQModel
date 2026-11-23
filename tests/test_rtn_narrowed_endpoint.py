# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""RTN must not narrow a finite grid reconstruction to ``inf``.

``RTN.quantize`` quantizes an FP32 copy of the weight and then narrows the
reconstruction back to the source dtype for replay and packing.  The symmetric
grid stores ``zero = 2 ** (bits - 1)`` and ``scale = 2 * amax / (2**bits - 1)``,
so the negative endpoint of a group rounds to code ``-2 ** (bits - 1)`` and
reconstructs ``2**bits * amax / (2**bits - 1)``, which is strictly larger than
``amax``.  For an FP16 weight with ``amax`` at the dtype maximum that
reconstruction is outside the finite range and the returned replay weight is
``-inf`` even though the source weight was finite.

The grid and its ties-to-even rounding are correct; the missing constraint is
that the codes a grid may select have to stay representable in the dtype the
reconstruction is narrowed to.  These tests drive the real ``RTN.quantize``
path and assert the returned weight is finite, keeps the source dtype, and
packs back to the codes that were scored.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import FORMAT, RTNConfig
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.quantization.rtn import RTN


WIDTH = 32
GROUP_SIZE = 32
FP16_MAX = torch.finfo(torch.float16).max


def _symmetric_weight(magnitude: float, dtype: torch.dtype) -> torch.Tensor:
    """Row-wise ``[-a, a, 0, ...]`` groups: one lane sets the fitted range."""

    weight = torch.zeros(WIDTH, WIDTH, dtype=dtype)
    weight[:, 0] = -magnitude
    weight[:, 1] = magnitude
    return weight


def _rtn_config(bits: int = 4, *, sym: bool = True) -> RTNConfig:
    return RTNConfig(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=sym,
        format=FORMAT.GPTQ,
        desc_act=False,
        mse=None,
        gsq=None,
        offload_to_disk=False,
    )


def _quantize(weight: torch.Tensor, *, bits: int = 4, sym: bool = True):
    layer = nn.Linear(WIDTH, WIDTH, bias=False, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    result = RTN(layer, _rtn_config(bits, sym=sym)).quantize()
    return layer, result


@pytest.mark.parametrize("magnitude", [60000.0, 61408.0, 61440.0, 65504.0])
def test_fp16_symmetric_endpoint_stays_finite(magnitude):
    weight = _symmetric_weight(magnitude, torch.float16)
    layer, (quantized, scale, zero, *_rest) = _quantize(weight)

    # The source parameter is never rewritten by this path.
    assert layer.weight.data.dtype == torch.float16
    assert torch.isfinite(layer.weight.data).all()

    assert quantized.dtype == torch.float16
    assert quantized.shape == weight.shape
    assert torch.isfinite(quantized).all(), (
        f"a={magnitude} narrowed to {int((~torch.isfinite(quantized)).sum())} "
        f"non-finite elements at the negative endpoint"
    )
    assert torch.isfinite(scale).all()
    assert torch.equal(zero, torch.full_like(zero, 2 ** 3))


@pytest.mark.parametrize("magnitude", [61408.0, 61440.0])
def test_adjacent_fp16_values_keep_the_reachable_endpoint(magnitude):
    """The two neighbouring FP16 values must not both lose the endpoint code."""

    quantized, scale, *_rest = _quantize(_symmetric_weight(magnitude, torch.float16))[1]
    assert torch.isfinite(quantized).all()
    # 61408 still reaches the extreme representable value; 61440 cannot, so the
    # endpoint code moves one step in and reconstructs a finite value.
    assert quantized[0, 0].item() == (-FP16_MAX if magnitude == 61408.0 else -57344.0)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_every_bit_width_stays_finite_at_the_fp16_extremes(bits):
    for magnitude in (FP16_MAX, -FP16_MAX):
        quantized, scale, zero, *_rest = _quantize(
            _symmetric_weight(abs(magnitude), torch.float16), bits=bits
        )[1]
        assert torch.isfinite(quantized).all(), f"bits={bits} a={magnitude}"
        assert torch.isfinite(scale).all()
        assert torch.equal(zero, torch.full_like(zero, 2 ** (bits - 1)))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_fp16_extreme_endpoint_is_representable_after_narrowing(bits):
    """Every returned magnitude must fit the source dtype's finite range."""

    quantized, *_rest = _quantize(_symmetric_weight(FP16_MAX, torch.float16), bits=bits)[1]
    assert quantized.abs().max().item() <= FP16_MAX


def test_normal_fp16_range_is_unchanged():
    """A grid whose endpoints already fit must keep its exact reconstruction."""

    quantized, scale, zero, *_rest = _quantize(
        _symmetric_weight(1000.0, torch.float16)
    )[1]
    # The symmetric INT4 grid is ``zero = 8`` with ``scale = 2 * amax / 15``.
    expected_scale = torch.tensor(2.0 * 1000.0 / 15.0, dtype=torch.float32)
    assert scale.dtype == torch.float32
    assert scale.flatten()[0].item() == expected_scale.item()
    assert zero.flatten()[0].item() == 8.0
    # The negative endpoint still rounds to code 0, and the small grid stays
    # bitwise-identical to a plain FP32 narrowing of the same reconstruction.
    expected = torch.tensor(-(16 / 15) * 1000.0, dtype=torch.float16).item()
    assert quantized[:, 0].tolist() == [expected] * WIDTH


def test_bf16_endpoint_stays_finite():
    """BF16 has the same grid shape with a different finite maximum."""

    quantized, scale, *_rest = _quantize(
        _symmetric_weight(torch.finfo(torch.bfloat16).max, torch.bfloat16)
    )[1]
    assert quantized.dtype == torch.bfloat16
    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()


@pytest.mark.parametrize("sym", [True, False])
def test_both_endpoints_at_the_fp16_limit_stay_finite(sym):
    """An affine group spanning the full finite range overflows on each end.

    ``zero = round(-xmin / scale)`` puts the endpoints at codes 0 and 15, so
    both reconstructions reach ``maxq * scale``; that product is the dtype
    maximum plus one rounding step and is not representable.
    """

    weight = torch.zeros(WIDTH, WIDTH, dtype=torch.float16)
    weight[:, 0] = -FP16_MAX
    weight[:, 1] = FP16_MAX
    quantized, scale, zero, *_rest = _quantize(weight, sym=sym)[1]

    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    assert quantized.abs().max().item() <= FP16_MAX
    # Both endpoints moved one code in when the full span could not be stored.
    assert quantized[0, 0].item() < 0
    assert quantized[0, 1].item() > 0


def test_asymmetric_controls_keep_their_endpoint():
    """One-sided and modest asymmetric groups keep their reachable endpoint."""

    negative_only = torch.zeros(WIDTH, WIDTH, dtype=torch.float16)
    negative_only[:, 0] = -FP16_MAX
    negative_only[:, 1:] = -1.0
    quantized, *_rest = _quantize(negative_only, sym=False)[1]
    assert torch.isfinite(quantized).all()
    assert quantized[0, 0].item() == -FP16_MAX

    modest = torch.zeros(WIDTH, WIDTH, dtype=torch.float16)
    modest[:, 0] = -10.0
    modest[:, 1] = 12.0
    modest[:, 2:] = 1.0
    quantized, scale, zero, *_rest = _quantize(modest, sym=False)[1]
    assert torch.isfinite(quantized).all()
    assert scale.flatten()[0].item() == pytest.approx(22.0 / 15.0, rel=0, abs=1e-6)
    # The existing clipping is untouched: endpoints round to their own codes.
    assert quantized[0, 0].item() == pytest.approx(-10.2666015625, rel=0, abs=1e-3)
    assert quantized[0, 1].item() == pytest.approx(11.7333984375, rel=0, abs=1e-3)


def test_narrowed_codes_match_the_packer_recovery():
    """The packer re-derives codes from ``Q``; the fix must keep them equal."""

    weight = _symmetric_weight(61440.0, torch.float16)
    quantized, scale, zero, g_idx, *_rest = _quantize(weight)[1]

    # The packer's formula, in FP32, on the narrowed replay weight.
    codes = torch.clamp(
        torch.round(quantized.float() / scale.float()) + zero.float(), 0, 15
    ).to(torch.int64)

    # Independently derive the finite endpoint code from the fitted scale: the
    # largest negative step whose reconstruction fits FP16 is ``floor(M/s)``.
    normalized = scale.float().flatten()[0]
    steps = int(torch.floor(torch.tensor(FP16_MAX) / normalized).item())
    if normalized * steps > FP16_MAX:
        steps -= 1
    endpoint_code = 8 - steps
    assert endpoint_code == 1

    assert codes[:, 0].tolist() == [endpoint_code] * WIDTH
    assert codes[:, 1].tolist() == [8 + steps] * WIDTH
    # Every recovered code reconstructs inside the dtype's finite range.
    assert (quantized.float().abs() <= FP16_MAX).all()
    assert g_idx.numel() == WIDTH


def test_next_layer_calibration_input_stays_finite():
    """The replay weight feeds the next layer's capture; it must stay finite."""

    torch.manual_seed(5997)
    first = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float16)
    with torch.no_grad():
        first.weight.copy_(_symmetric_weight(61440.0, torch.float16))
    second = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.float16)
    with torch.no_grad():
        second.weight.copy_(torch.eye(WIDTH, dtype=torch.float16))

    quantized, *_rest = RTN(first, _rtn_config()).quantize()
    first.weight.data = quantized

    activations = torch.full((4, WIDTH), 0.5, dtype=torch.float16)
    captured = first(activations)
    assert torch.isfinite(captured).all()
    assert torch.isfinite(second(captured)).all()


def test_quantizer_value_limit_bounds_the_selected_codes():
    """The shared grid helper must be inert unless a limit is supplied."""

    x = _symmetric_weight(61440.0, torch.float16).reshape(WIDTH, WIDTH)

    unbounded = Quantizer(qcfg=_rtn_config())
    unbounded.configure(perchannel=True)
    unbounded.find_params(x, weight=True)
    assert unbounded.code_lo is None and unbounded.code_hi is None
    # Unbounded selection still reaches code 0 and its FP32 reconstruction is
    # finite; only narrowing it to FP16 overflows.
    assert torch.isfinite(unbounded.quantize(x)).all()
    assert not torch.isfinite(unbounded.quantize(x).half()).all()

    bounded = Quantizer(qcfg=_rtn_config())
    bounded.configure(perchannel=True)
    bounded.find_params(x, weight=True, value_limit=FP16_MAX)
    assert bounded.code_lo.flatten()[0].item() == 1
    assert bounded.code_hi.flatten()[0].item() == 15
    assert torch.isfinite(bounded.quantize(x).half()).all()


def test_mse_search_scores_the_narrowed_codes():
    """SmoothMSE picks its winner on the codes that survive narrowing."""

    from gptqmodel.quantization.config import MSEConfig

    x = _symmetric_weight(61440.0, torch.float16).reshape(WIDTH, WIDTH)

    qcfg = _rtn_config()
    qcfg.mse = MSEConfig(norm=2.0)
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    quantizer.find_params(x, weight=True, value_limit=FP16_MAX)

    assert torch.isfinite(quantizer.quantize(x)).all()
    assert quantizer.code_lo.flatten()[0].item() >= 1


def test_integer_source_dtype_keeps_the_unconstrained_grid():
    """Integer weights express range as counts, so no magnitude bound applies."""

    from gptqmodel.quantization.rtn import _finite_value_limit

    assert _finite_value_limit(torch.float16) == FP16_MAX
    assert _finite_value_limit(torch.bfloat16) == torch.finfo(torch.bfloat16).max
    assert _finite_value_limit(torch.float32) == torch.finfo(torch.float32).max
    assert _finite_value_limit(torch.int8) is None


def test_per_row_bounds_follow_each_lane():
    """A per-row bound must move only the lane it belongs to."""

    from gptqmodel.quantization.quantizer import _finite_code_bounds

    # Row 0 fits its endpoint; row 1 needs the code moved one step in.
    scale = torch.tensor([8000.0, 8192.0])
    zero = torch.full_like(scale, 8.0)
    limit = torch.tensor([FP16_MAX, FP16_MAX])

    lo, hi = _finite_code_bounds(scale, zero, 15, False, limit)
    assert lo.tolist() == [0.0, 1.0]
    assert hi.tolist() == [15.0, 15.0]


def test_bounds_follow_the_quantizer_device_and_stay_out_of_state():
    """``code_lo``/``code_hi`` mirror the grid through moves, not checkpoints."""

    x = _symmetric_weight(61440.0, torch.float16).reshape(WIDTH, WIDTH)
    quantizer = Quantizer(qcfg=_rtn_config())
    quantizer.configure(perchannel=True)
    quantizer.find_params(x, weight=True, value_limit=FP16_MAX)

    assert "code_lo" not in quantizer.state_dict()
    assert "code_hi" not in quantizer.state_dict()
    # A move must not detach the bound from the grid it describes.
    moved = quantizer.to("cpu")
    assert moved.code_lo.device == moved.scale.device
    assert moved.code_lo.dtype == moved.scale.dtype
    assert moved.code_lo.flatten()[0].item() == 1


def test_square_group_bound_stays_one_dimensional():
    """A ``[rows]`` bound on a square lane grid must not broadcast to a square."""

    from gptqmodel.quantization.quantizer import _finite_code_bounds

    scale = torch.full((WIDTH, WIDTH), 8192.0)
    zero = torch.full_like(scale, 8.0)
    lo, hi = _finite_code_bounds(scale, zero, 15, False, torch.full((WIDTH,), FP16_MAX))

    # ``[rows, rows]`` would have silently clamped with the wrong lane per code.
    assert lo.shape == scale.shape
    assert hi.shape == scale.shape
    assert lo.tolist() == [[1.0] * WIDTH for _ in range(WIDTH)]


@pytest.mark.parametrize("sym", [True, False])
def test_smooth_mse_candidates_score_finite_codes(sym):
    """The SmoothMSE ternary search must score representable endpoints."""

    from gptqmodel.quantization.config import MSEConfig, SmoothMSE

    qcfg = _rtn_config(sym=sym)
    qcfg.mse = MSEConfig(norm=2.0)
    qcfg.smooth = SmoothMSE(steps=16, maxshrink=0.8)

    weight = _symmetric_weight(FP16_MAX, torch.float16)
    layer = nn.Linear(WIDTH, WIDTH, bias=False, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    quantized, scale, zero, *_rest = RTN(layer, qcfg).quantize()

    assert quantized.dtype == torch.float16
    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    assert quantized.abs().max().item() <= FP16_MAX


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_packed_layer_dequantizes_to_the_narrowed_grid(bits):
    """The real packer must recover the same codes the quantizer selected."""

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    weight = _symmetric_weight(FP16_MAX, torch.float16)
    layer, (quantized, scale, zero, g_idx, *_rest) = _quantize(weight, bits=bits)
    layer.weight.data = quantized

    module = TorchLinear(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=WIDTH,
        out_features=WIDTH,
        bias=False,
        format=FORMAT.GPTQ,
        register_buffers=False,
    )
    module.pack_block(layer, scale.clone(), zero.clone(), g_idx.clone())

    # Same independent oracle the kernel-accuracy suite uses: recover codes
    # from the narrowed weight with the packer's formula, then rebuild.
    maxq = (1 << bits) - 1
    scale_full = scale[:, g_idx.long()]
    zero_full = zero[:, g_idx.long()]
    codes = torch.round(
        (quantized.float() + zero_full * scale_full) / scale_full
    ).clamp(0, maxq)
    reference = ((codes - zero_full) * scale_full.to(torch.float16).float()).T.contiguous()

    dequantized = module.dequantize_weight().float()
    assert torch.isfinite(dequantized).all()
    assert torch.allclose(dequantized, reference, atol=1e-3, rtol=0)
    assert dequantized.abs().max().item() <= FP16_MAX
