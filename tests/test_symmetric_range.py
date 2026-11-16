# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Symmetric range fitting must be centered on zero.

A symmetric checkpoint stores the fixed zero point ``(maxq + 1) // 2`` (8 for
INT4), so every group has to be fitted against ``[-max|x|, max|x|]``.  A group
whose values never go negative used to be fitted as ``[0, max|x|]`` while the
zero point stayed at 8.  The grid then covered 15 steps of ``max|x| / 15``
instead of 15 steps of ``2 * max|x| / 15``, so the positive half saturated at
``7/15`` of the true maximum.

The references below are written independently of ``find_params``: they use
their own reductions and hand-derived grid endpoints, so the tests do not
merely compare the implementation with itself.
"""

import pytest
import torch
from torch import nn

from gptqmodel.models._const import normalize_device
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import QQQConfig, QuantizeConfig, RTNConfig
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.quantization.qqq import Quantizer as QQQQuantizer
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.quantization.rtn import RTN
from gptqmodel.utils.backend import BACKEND


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
BITS = (2, 3, 4, 8)
ROWS = 4
WIDTH = 32
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _groups(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    ramp = torch.linspace(0.0, 1.0, WIDTH, dtype=torch.float64).repeat(ROWS, 1)
    descending = torch.linspace(0.0, -1.0, WIDTH, dtype=torch.float64).repeat(ROWS, 1)
    mixed = torch.linspace(-2.0, 3.0, WIDTH, dtype=torch.float64).repeat(ROWS, 1)
    mixed[0, ::4] = 0.0
    return {
        # Every value is strictly positive: the regression case.
        "all_positive": torch.ones(ROWS, WIDTH, dtype=dtype),
        "all_negative": -torch.ones(ROWS, WIDTH, dtype=dtype),
        # Minimum is exactly zero but the group is not constant.
        "positive_ramp_from_zero": ramp.to(dtype),
        "negative_ramp_to_zero": descending.to(dtype),
        # Crosses zero.
        "mixed_signs": mixed.to(dtype),
        # Constant non-unit groups.
        "constant_positive": torch.full((ROWS, WIDTH), 2.5, dtype=dtype),
        "constant_negative": torch.full((ROWS, WIDTH), -2.5, dtype=dtype),
        # Signed zero must still take the +-1 sentinel.
        "signed_zero": torch.full((ROWS, WIDTH), -0.0, dtype=dtype),
        "all_zero": torch.zeros(ROWS, WIDTH, dtype=dtype),
    }


GROUP_NAMES = list(_groups(torch.float32))


def _reference_symmetric_params(group: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-centered bound and scale for one group per row.

    ``find_params`` normalizes the reduction to float32 through its zero
    template, so the reference widens the same way and stays bit-comparable.
    ``maxq`` is a tensor here to match the quantizer's tensor/tensor division
    instead of the scalar fast path.
    """

    maxq = 2**bits - 1
    bound = group.float().abs().amax(dim=1)
    bound = torch.where(bound == 0, torch.ones_like(bound), bound)
    maxq_tensor = torch.tensor(maxq, dtype=torch.int64, device=group.device)
    scale = (bound + bound) / maxq_tensor
    return bound, scale


def _reference_dequant(group: torch.Tensor, bits: int) -> torch.Tensor:
    """Reconstruct through the reference grid, not through ``quantize``."""

    maxq = 2**bits - 1
    zero = (maxq + 1) // 2
    _, scale = _reference_symmetric_params(group, bits)
    codes = torch.clamp(
        torch.round(group.float() / scale.unsqueeze(1)) + zero, 0, maxq
    )
    return (codes - zero) * scale.unsqueeze(1)


def _find_params(group: torch.Tensor, bits: int, device: str) -> Quantizer:
    qcfg = QuantizeConfig(bits=bits, group_size=-1, sym=True, desc_act=False)
    quantizer = Quantizer(qcfg).to(device)
    quantizer.configure(perchannel=True, bits=bits, sym=True)
    quantizer.find_params(group.to(device), weight=True)
    return quantizer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("name", GROUP_NAMES)
def test_symmetric_range_is_zero_centered(dtype, bits, name, device):
    """Every group must fit ``[-max|x|, max|x|]`` against the fixed midpoint."""

    group = _groups(dtype)[name]
    quantizer = _find_params(group, bits, device)

    bound, scale = _reference_symmetric_params(group.to(device), bits)
    fitted_scale = quantizer.scale.flatten()
    fitted_zero = quantizer.zero.flatten()

    assert torch.equal(fitted_scale, scale), (name, fitted_scale, scale)
    assert torch.all(fitted_zero == (2**bits) // 2), (name, fitted_zero)
    # ``scale * maxq`` must span both halves of the range.  A one-sided fit
    # leaves the lower bound at zero and spans only ``max|x|``.
    assert torch.allclose(
        fitted_scale.double() * (2**bits - 1), 2.0 * bound.double(), rtol=1e-6, atol=0
    ), (name, fitted_scale, bound)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "name",
    [
        "all_positive",
        "all_negative",
        "positive_ramp_from_zero",
        "negative_ramp_to_zero",
        "mixed_signs",
        "constant_positive",
        "constant_negative",
        "all_zero",
    ],
)
def test_symmetric_dequant_matches_independent_reconstruction(dtype, name, device):
    """Dequantization must match the reference grid evaluation."""

    group = _groups(dtype)[name].to(device)
    quantizer = _find_params(group, 4, device)

    actual = quantizer.quantize(group)
    expected = _reference_dequant(group, 4)
    assert torch.equal(actual, expected), (name, actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "bits,expected_fraction",
    [(2, 2.0 / 3.0), (4, 14.0 / 15.0), (8, 254.0 / 255.0)],
)
def test_all_positive_group_reaches_the_top_code(dtype, bits, expected_fraction):
    """A +1 group must decode to ``(maxq - zero) * scale``, not half of it."""

    group = torch.ones(ROWS, WIDTH, dtype=dtype)
    quantizer = _find_params(group, bits, "cpu")
    dequant = quantizer.quantize(group)

    assert torch.allclose(
        dequant.float(),
        torch.full_like(dequant.float(), expected_fraction),
        rtol=0,
        atol=1e-6,
    ), dequant
    # The pre-fix one-sided range saturated here at 7/15 (0.4666 for INT4).
    assert float(dequant.max()) > 0.6


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("method", ["gptq", "qqq"])
def test_non_negative_fit_rounding_neighbors_and_saturation(device, dtype, bits, method):
    # This bound makes the corrected scale exactly one, even in BF16. The
    # expected codes can therefore be derived in FP64 without rounding ties
    # introduced by scale fitting or reciprocal multiplication.
    maxq = 2**bits - 1
    midpoint = 2 ** (bits - 1)
    bound = maxq / 2
    group = torch.full((ROWS, WIDTH), bound, dtype=dtype, device=device)
    if method == "gptq":
        quantizer = _find_params(group, bits, device)
    else:
        quantizer = QQQQuantizer().to(device)
        quantizer.configure(bits=bits, perchannel=True, sym=True, groupsize=WIDTH)
        quantizer.find_params(group, weight=True)
    assert torch.equal(quantizer.scale, torch.ones_like(quantizer.scale))
    assert torch.all(quantizer.zero == midpoint)

    half = torch.tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=dtype)
    neighbors = torch.stack((
        torch.nextafter(half, torch.full_like(half, -float("inf"))),
        half,
        torch.nextafter(half, torch.full_like(half, float("inf"))),
    ), dim=1).flatten()
    endpoints = torch.tensor([-2 * bound, -bound, -0.0, 0.0, bound, 2 * bound], dtype=dtype)
    probes = torch.cat((neighbors, endpoints)).repeat(ROWS, 1).to(device)
    expected_codes = (probes.double().round() + midpoint).clamp(0, maxq)
    actual = quantizer.quantize(probes)
    assert torch.equal(actual, (expected_codes - midpoint).float())
    assert torch.equal((actual.double() + midpoint).to(torch.int64), expected_codes.to(torch.int64))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "bits,expected_positive",
    [
        (2, 2.0),  # (maxq - zero) * scale with bound 3.0
        (4, 2.8),
        (8, 3.0 * 127.0 / 127.5),
    ],
)
def test_symmetric_grid_covers_both_extremes(dtype, bits, expected_positive):
    """Both signs must land within one grid step of the source extremes."""

    bound = 3.0
    group = torch.tensor([[bound, -bound] * (WIDTH // 2)], dtype=dtype)
    quantizer = _find_params(group, bits, "cpu")
    scale = float(quantizer.scale.flatten()[0])
    dequant = quantizer.quantize(group.to("cpu"))

    positive = float(dequant[0, 0])
    negative = float(dequant[0, 1])
    # A one-sided range would decode +bound to 7/15 of it.
    assert positive == pytest.approx(expected_positive, rel=0, abs=1e-6)
    assert abs(positive - bound) <= scale + 1e-6, (positive, scale)
    assert abs(negative + bound) <= scale + 1e-6, (negative, scale)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", BITS)
def test_rtn_all_positive_group_packed_decode(dtype, bits, device):
    """Check real RTN output and complete packed bytes against a golden grid."""

    out_features, in_features, group_size = 32, 64, 32
    maxq, midpoint = 2**bits - 1, 2 ** (bits - 1)
    layer = nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)
    layer.weight.data.fill_(1.0)

    qcfg = RTNConfig(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        offload_to_disk=False,
        device=normalize_device(device),
    )
    quantized, scale, zero, g_idx, *_ = RTN(layer, qcfg=qcfg).quantize()

    expected_scale = torch.tensor(2.0, device=device) / torch.tensor(maxq, device=device)
    expected_weight = ((maxq - midpoint) * expected_scale).to(dtype)
    assert torch.all(zero == midpoint)
    torch.testing.assert_close(scale, expected_scale.expand_as(scale), rtol=0, atol=0)
    torch.testing.assert_close(quantized, expected_weight.expand_as(quantized), rtol=0, atol=0)

    packed = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        backend=BACKEND.TORCH,
        bias=False,
    )
    packed.pack_block(
        layer.to("cpu"), scale.cpu(), zero.cpu(), g_idx=g_idx.to(device="cpu", dtype=torch.int32)
    )
    # Every weight must use the top code. A continuous little-endian stream
    # also covers three-bit codes that straddle int32 words.
    assert torch.equal(packed.qweight, torch.full_like(packed.qweight, -1))
    zero_stream = sum(midpoint << (bits * index) for index in range(out_features))
    zero_words = [
        ((zero_stream >> (32 * index)) + 2**31) % 2**32 - 2**31
        for index in range(bits)
    ]
    expected_qzeros = torch.tensor(zero_words, dtype=torch.int32).repeat(in_features // group_size, 1)
    assert torch.equal(packed.qzeros, expected_qzeros)
    expected_stored_scale = torch.full(packed.scales.shape, 2.0 / maxq, dtype=torch.float16)
    assert torch.equal(packed.scales, expected_stored_scale)
    expected_g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    assert torch.equal(packed.g_idx, expected_g_idx)
    packed.post_init()

    decoded = packed.dequantize_weight().T
    # Stored scales and visible weights each have their own dtype rounding.
    expected_decoded = torch.full_like(decoded, 2.0 / maxq) * (maxq - midpoint)
    torch.testing.assert_close(decoded, expected_decoded, rtol=0, atol=0)
    roundtrip_atol = max(float(torch.finfo(dtype).eps), float(torch.finfo(decoded.dtype).eps))
    torch.testing.assert_close(
        decoded, quantized.to(device=decoded.device, dtype=decoded.dtype), rtol=0, atol=roundtrip_atol
    )


def test_qqq_quantizer_all_positive_group_is_zero_centered():
    """QQQ's grouped affine path uses the same fixed midpoint."""

    quantizer = QQQQuantizer()
    quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=128, mse=False)
    group = torch.ones(2, 128)
    quantizer.find_params(group, weight=True)

    assert torch.all(quantizer.zero == 8), quantizer.zero
    torch.testing.assert_close(
        quantizer.scale.flatten(),
        torch.full((2,), 2.0 / 15.0),
        rtol=0,
        atol=1e-6,
    )
    dequant = quantizer.quantize(group)
    torch.testing.assert_close(
        dequant, torch.full_like(dequant, 14.0 / 15.0), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_qqq_grouped_quantize_reconstructs_positive_groups(dtype, device):
    """The real QQQ grouped path must not saturate an all-ones group at 7/15."""

    torch.manual_seed(0)
    layer = nn.Linear(128, 8, bias=False, dtype=dtype, device=device)
    layer.weight.data.fill_(1.0)
    task = QQQ(
        layer,
        QQQConfig(bits=4, group_size=128, desc_act=False, device=normalize_device(device)),
    )
    task.quantizer.configure(
        4, perchannel=True, sym=True, mse=False, norm=2.4, groupsize=128
    )
    task.add_batch(torch.randn(4, 128, dtype=dtype, device=device), None)

    weight, scale, zero = task.quantize()[:3]

    assert torch.all(zero == 8), zero
    assert float(weight.max()) > 0.9, weight.max()
    assert float(weight.max()) == pytest.approx(14.0 / 15.0, rel=0, abs=1e-3)
