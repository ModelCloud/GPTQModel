# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch-oracle checks for native MLX FOEM quantization."""

import sys
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from gptqmodel.quantization import foem as foem_module
from gptqmodel.quantization.config import FOEMConfig, GPTQConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.mlx_foem import foem_quantize_weight_mlx


def _torch_params(group, bits, sym):
    """Reproduce Quantizer.find_params without calling the implementation."""
    minimum = torch.minimum(group.amin(dim=1), torch.zeros(group.shape[0]))
    maximum = torch.maximum(group.amax(dim=1), torch.zeros(group.shape[0]))
    if sym:
        maximum = torch.maximum(minimum.abs(), maximum)
        minimum = torch.where(minimum < 0, -maximum, minimum)
    empty = (minimum == 0) & (maximum == 0)
    minimum = torch.where(empty, -1, minimum)
    maximum = torch.where(empty, 1, maximum)
    scale = (maximum - minimum) / (2**bits - 1)
    zero = (
        torch.full_like(scale, 2 ** (bits - 1))
        if sym
        else torch.round(-minimum / scale)
    )
    return scale, zero


def _torch_foem_oracle(
    weight,
    inverse_hessian,
    bits,
    group_size,
    beta,
    sym,
    *,
    return_residuals=False,
):
    """Follow FOEM's sequential column corrections using independent Torch ops."""
    raw = torch.from_numpy(weight.copy())
    factor = torch.from_numpy(inverse_hessian.copy())
    _, columns = raw.shape
    remaining = raw.clone()
    quantized, scales, zeros = [], [], []
    residuals = torch.empty_like(raw) if return_residuals else None
    for start in range(0, columns, group_size):
        end = start + group_size
        group = remaining[:, :group_size].clone()
        scale, zero = _torch_params(group, bits, sym)
        errors = torch.empty_like(group)
        output = torch.empty_like(group)
        for offset in range(group_size):
            value = group[:, offset].clone()
            scaled = value / scale
            code = (torch.round(scaled) + zero).clamp(0, 2**bits - 1)
            q = scale * (code - zero)
            if return_residuals:
                residuals[:, start + offset] = value - q
            error = ((value - q) - (value - raw[:, start + offset]) * beta) / factor[
                start + offset, start + offset
            ]
            output[:, offset] = q
            errors[:, offset] = error
            group[:, offset:] -= (
                error[:, None] * factor[start + offset, start + offset : end]
            )
            if offset + 1 < group_size:
                group[:, offset + 1] -= beta * (
                    group[:, offset + 1] - raw[:, start + offset + 1]
                )
        quantized.append(output)
        scales.append(scale[:, None])
        zeros.append(zero[:, None])
        if end < columns:
            remaining = remaining[:, group_size:] - errors @ factor[start:end, end:]
    result = tuple(
        torch.cat(values, dim=1).numpy() for values in (quantized, scales, zeros)
    )
    if return_residuals:
        result = (*result, residuals.numpy())
    return result


def _processor_quantize(source, factor, config, *, blocksize):
    layer = torch.nn.Linear(
        source.shape[1], source.shape[0], bias=False, dtype=source.dtype
    )
    layer.weight.data.copy_(source)
    task = FOEM(layer, config)
    task.quantizer.configure(
        perchannel=True,
        grid=100,
        maxshrink=0.8,
    )
    task.H = torch.eye(source.shape[1], dtype=torch.float32, device=source.device)
    task.nsamples = 4
    task.hessian_inverse = lambda _hessian: (factor.clone(), config.damp_percent)
    return task.quantize(blocksize=blocksize)


def _adjudicate_banded_foem_row(
    source_row, expected_codes, actual_codes, group_size=128
):
    """Classify direct exact ties and differences propagated from those ties."""
    mismatch_columns = np.flatnonzero(actual_codes != expected_codes)
    if not len(mismatch_columns):
        return set(), set()

    def trace(codes):
        direct_ties = set()
        propagated = set()
        with localcontext() as context:
            context.prec = 80
            raw = [Decimal.from_float(float(value)) for value in source_row]
            working = raw.copy()
            upper = Decimal.from_float(float(np.float32(0.05)))
            beta = Decimal("0.2")
            prior_direct_tie = False
            last = int(mismatch_columns[-1])
            for start in range(0, last + 1, group_size):
                end = min(start + group_size, len(raw))
                group = working[start:end]
                maximum = max(
                    abs(min(Decimal(0), min(group))), max(Decimal(0), max(group))
                )
                scale = 2 * maximum / 15
                for column in range(start, min(end, last + 1)):
                    value = working[column]
                    quotient = value / scale
                    fraction = quotient - quotient.to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                    margin = abs(fraction - Decimal("0.5"))
                    nearest = max(
                        0,
                        min(
                            15,
                            int(quotient.to_integral_value(rounding=ROUND_HALF_EVEN))
                            + 8,
                        ),
                    )
                    observed = int(codes[column])
                    if column in mismatch_columns:
                        if margin < Decimal("1e-12") and abs(observed - nearest) <= 1:
                            direct_ties.add(column)
                            prior_direct_tie = True
                        elif prior_direct_tie and observed == nearest:
                            propagated.add(column)
                        elif observed != nearest:
                            raise AssertionError(
                                f"column {column} is {margin} code units from a tie; "
                                f"high precision selects {nearest}, observed {observed}"
                            )
                    quantized = scale * (observed - 8)
                    error = (value - quantized) - (value - raw[column]) * beta
                    if column + 1 < len(working):
                        working[column + 1] -= error * upper
                    if column + 1 < end:
                        working[column + 1] -= (
                            working[column + 1] - raw[column + 1]
                        ) * beta
        return direct_ties, propagated

    expected_direct, expected_propagated = trace(expected_codes)
    actual_direct, actual_propagated = trace(actual_codes)
    direct = expected_direct | actual_direct
    propagated = (expected_propagated | actual_propagated) - direct
    return direct, propagated


def _codes_from_dequantized(result, group_size):
    quantized, scales, zeros = result
    rows, columns = quantized.shape
    codes = np.rint(
        quantized.reshape(rows, columns // group_size, group_size) / scales[:, :, None]
        + zeros[:, :, None]
    )
    return codes.astype(np.uint8)


@pytest.mark.parametrize(
    "bits,sym",
    [(bits, sym) for bits in range(2, 9) for sym in (True, False)],
)
@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("beta", [0.0, 0.2])
def test_foem_group_updates_match_torch(bits, sym, group_size, beta):
    rng = np.random.default_rng(600 + bits + group_size)
    rows, columns = 7, group_size * 2
    weight = rng.normal(0, 0.3, (rows, columns)).astype(np.float32)
    factor = np.eye(columns, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.002, (columns, columns)).astype(np.float32), 1)
    expected = _torch_foem_oracle(weight, factor, bits, group_size, beta, sym)
    actual = foem_quantize_weight_mlx(
        mx.array(weight),
        mx.array(factor),
        bits=bits,
        group_size=group_size,
        beta=beta,
        sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6
        )
    np.testing.assert_array_equal(
        _codes_from_dequantized(
            tuple(np.asarray(value) for value in actual), group_size
        ),
        _codes_from_dequantized(expected, group_size),
    )


@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("sym", [True, False])
def test_foem_code_boundaries_one_float32_step(bits, sym):
    weight = np.zeros((4, 32), dtype=np.float32)
    weight[:, 0], weight[:, 1] = -1, 1
    scale = np.float32(2 / (2**bits - 1))
    midpoint = np.float32(scale / 2)
    weight[0, 2:5] = [
        np.nextafter(midpoint, np.float32(-np.inf)),
        midpoint,
        np.nextafter(midpoint, np.float32(np.inf)),
    ]
    weight[1, 2:5] = -weight[0, 2:5]
    weight[2, 2:5] = [0, -0.0, 1]
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(weight, factor, bits, 32, 0.2, sym)
    actual = foem_quantize_weight_mlx(
        mx.array(weight),
        mx.array(factor),
        bits=bits,
        group_size=32,
        sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6
        )
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 32),
        _codes_from_dequantized(expected, 32),
    )


def test_foem_asymmetric_zero_point_tie_and_neighbors():
    weight = np.zeros((3, 32), dtype=np.float32)
    minimum = np.float32(-17)
    weight[0, 0], weight[0, 1] = minimum, 13
    weight[1, 0], weight[1, 1] = np.nextafter(minimum, np.float32(-np.inf)), 13
    weight[2, 0], weight[2, 1] = np.nextafter(minimum, np.float32(np.inf)), 13
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(weight, factor, 4, 32, 0.2, False)
    actual = foem_quantize_weight_mlx(
        mx.array(weight),
        mx.array(factor),
        group_size=32,
        sym=False,
    )
    assert expected[2][0, 0] == 8  # 8.5 rounds to even.
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize("sym", [True, False])
def test_foem_fused_params_extrema_match_torch(sym):
    weight = np.zeros((4, 32), dtype=np.float32)
    weight[1] = np.linspace(0.125, 2, 32, dtype=np.float32)
    weight[2] = -weight[1]
    weight[3, :2] = (-3, 2)
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(weight, factor, 4, 32, 0.2, sym)
    actual = foem_quantize_weight_mlx(
        mx.array(weight),
        mx.array(factor),
        group_size=32,
        sym=sym,
    )
    for index in range(3):
        np.testing.assert_array_equal(np.asarray(actual[index]), expected[index])


@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [
        (mx.float16, torch.float16),
        (mx.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("sym", [True, False])
def test_foem_low_precision_code_midpoint(dtype, torch_dtype, sym):
    midpoint = torch.tensor(1 / 15, dtype=torch_dtype)
    lower = torch.nextafter(midpoint, torch.tensor(-float("inf"), dtype=torch_dtype))
    upper = torch.nextafter(midpoint, torch.tensor(float("inf"), dtype=torch_dtype))
    source = np.zeros((2, 32), dtype=np.float32)
    source[:, 0], source[:, 1] = -1, 1
    source[0, 2:5] = torch.stack((lower, midpoint, upper)).float().numpy()
    source[1, 2:5] = -source[0, 2:5]
    weight = mx.array(source).astype(dtype)
    factor = np.eye(32, dtype=np.float32)
    expected = _torch_foem_oracle(
        np.asarray(weight.astype(mx.float32)),
        factor,
        4,
        32,
        0.2,
        sym,
    )
    actual = foem_quantize_weight_mlx(weight, mx.array(factor), group_size=32, sym=sym)
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6
        )
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 32),
        _codes_from_dequantized(expected, 32),
    )


@pytest.mark.parametrize("sym", [True, False])
def test_foem_bfloat16_with_cross_group_updates(sym):
    rng = np.random.default_rng(7900)
    source = rng.normal(0, 0.3, (9, 128)).astype(np.float32)
    weight = mx.array(source).astype(mx.bfloat16)
    mx.eval(weight)
    factor = np.eye(128, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.002, (128, 128)).astype(np.float32), 1)
    expected = _torch_foem_oracle(
        np.asarray(weight.astype(mx.float32)),
        factor,
        4,
        64,
        0.2,
        sym,
    )
    actual = foem_quantize_weight_mlx(
        weight,
        mx.array(factor),
        bits=4,
        group_size=64,
        beta=0.2,
        sym=sym,
    )
    for index in range(3):
        np.testing.assert_allclose(
            np.asarray(actual[index]), expected[index], rtol=1e-6, atol=1e-6
        )
    np.testing.assert_array_equal(
        _codes_from_dequantized(tuple(np.asarray(value) for value in actual), 64),
        _codes_from_dequantized(expected, 64),
    )


def test_foem_returned_residuals_match_independent_torch_oracle():
    rng = np.random.default_rng(8124)
    weight = rng.normal(0, 0.2, (9, 128)).astype(np.float32)
    factor = np.eye(128, dtype=np.float32)
    factor += np.triu(rng.normal(0, 0.001, (128, 128)).astype(np.float32), 1)
    expected = _torch_foem_oracle(
        weight, factor, 4, 64, 0.2, True, return_residuals=True
    )
    actual = foem_quantize_weight_mlx(
        mx.array(weight),
        mx.array(factor),
        bits=4,
        group_size=64,
        beta=0.2,
        sym=True,
        return_residuals=True,
    )
    assert len(actual) == 4
    np.testing.assert_allclose(np.asarray(actual[3]), expected[3], rtol=0, atol=1e-6)
    diagonal = torch.from_numpy(np.diag(factor).copy())
    actual_residuals = torch.from_numpy(np.asarray(actual[3]))
    expected_residuals = torch.from_numpy(expected[3])
    actual_loss = torch.sum(
        actual_residuals.square() / diagonal.square() / 2, dtype=torch.float64
    )
    expected_loss = torch.sum(
        expected_residuals.square() / diagonal.square() / 2, dtype=torch.float64
    )
    assert abs(float(actual_loss) - float(expected_loss)) <= 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_foem_processor_mlx_matches_torch_oracle(monkeypatch, dtype):
    generator = torch.Generator().manual_seed(8125)
    source = torch.randn((7, 128), generator=generator, dtype=torch.float32).to(dtype)
    factor = torch.eye(128, dtype=torch.float32)
    factor.diagonal(offset=1).fill_(0.025)
    config = GPTQConfig(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        mse=0,
        foem=FOEMConfig(alpha=0, beta=0.2),
    )

    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: False)
    expected = _processor_quantize(source, factor, config, blocksize=64)

    calls = 0
    original = foem_module._quantize_foem_weight_mlx_to_torch

    def counted_mlx(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: True)
    monkeypatch.setattr(foem_module, "_MLX_FOEM_MIN_ELEMENTS", 0)
    monkeypatch.setattr(foem_module, "_quantize_foem_weight_mlx_to_torch", counted_mlx)
    actual = _processor_quantize(source, factor, config, blocksize=64)

    assert calls == 1
    for index in (0, 1, 2):
        torch.testing.assert_close(
            actual[index].float(), expected[index].float(), rtol=0, atol=1e-6
        )
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    assert abs(actual[5] - expected[5]) <= 1e-6
    assert actual[6:] == expected[6:]


def test_foem_processor_keeps_torch_below_mlx_crossover(monkeypatch):
    source = torch.zeros((2, 32), dtype=torch.float16)
    factor = torch.eye(32, dtype=torch.float32)
    config = GPTQConfig(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        foem=FOEMConfig(alpha=0, beta=0.2),
    )
    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: True)
    monkeypatch.setattr(foem_module, "_MLX_FOEM_MIN_ELEMENTS", source.numel() + 1)

    def reject_mlx(*args, **kwargs):
        raise AssertionError("small FOEM weight used the slower MLX path")

    monkeypatch.setattr(foem_module, "_quantize_foem_weight_mlx_to_torch", reject_mlx)
    _processor_quantize(source, factor, config, blocksize=32)


@pytest.mark.parametrize(
    "change",
    [
        {"alpha": 0.25},
        {"desc_act": True},
        {"static_groups": True},
        {"mse": 1.0},
        {"blocksize": 64},
    ],
)
def test_foem_processor_rejects_unsupported_mlx_modes(monkeypatch, change):
    config = GPTQConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=change.get("desc_act", False),
        act_group_aware=False,
        static_groups=change.get("static_groups", False),
        mse=change.get("mse", 0),
        foem=FOEMConfig(alpha=change.get("alpha", 0), beta=0.2),
    )
    weight = torch.zeros((2, 256), dtype=torch.float16)
    factor = torch.eye(256, dtype=torch.float32)
    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: True)
    monkeypatch.setattr(foem_module, "_MLX_FOEM_MIN_ELEMENTS", 0)
    assert not foem_module._should_use_mlx_foem_quantization(
        weight, factor, config, blocksize=change.get("blocksize", 128)
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires a Torch MPS device"
)
def test_foem_processor_mps_bridge_matches_cpu_torch_oracle(monkeypatch):
    generator = torch.Generator().manual_seed(8126)
    source = torch.randn((7, 128), generator=generator, dtype=torch.float16)
    factor = torch.eye(128, dtype=torch.float32)
    factor.diagonal(offset=1).fill_(0.025)
    config = GPTQConfig(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        foem=FOEMConfig(alpha=0, beta=0.2),
    )
    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: False)
    expected = _processor_quantize(source, factor, config, blocksize=64)

    monkeypatch.setattr(foem_module, "_mlx_foem_quantization_available", lambda: True)
    monkeypatch.setattr(foem_module, "_MLX_FOEM_MIN_ELEMENTS", 0)
    actual = _processor_quantize(
        source.to("mps"), factor.to("mps"), config, blocksize=64
    )

    for index in (0, 1, 2, 3):
        torch.testing.assert_close(
            actual[index].cpu().float(), expected[index].float(), rtol=0, atol=1e-6
        )
    assert abs(actual[5] - expected[5]) <= 1e-6


@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
def test_foem_qwen38_27b_full_projection(name, out_features, in_features, dtype):
    """Compare every low-precision output and code with Hessian corrections."""
    del name
    group_size, bits = 128, 4
    rng = np.random.default_rng(7700 + out_features + in_features)
    source = rng.normal(0, 0.2, (out_features, in_features)).astype(np.float32)
    weight = mx.array(source).astype(dtype)
    mx.eval(weight)
    source = np.asarray(weight.astype(mx.float32))
    factor = np.eye(in_features, dtype=np.float32)
    np.fill_diagonal(factor[:, 1:], 0.05)
    expected = _torch_foem_oracle(source, factor, bits, group_size, 0.2, True)
    actual = foem_quantize_weight_mlx(
        weight,
        mx.array(factor),
        bits=bits,
        group_size=group_size,
        beta=0.2,
        sym=True,
    )
    output = np.asarray(actual[0])
    np.testing.assert_allclose(np.asarray(actual[1]), expected[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[2]), expected[2])
    actual_codes = _codes_from_dequantized(
        tuple(np.asarray(value) for value in actual), group_size
    ).reshape(output.shape)
    expected_codes = _codes_from_dequantized(expected[:3], group_size).reshape(
        output.shape
    )
    accepted = np.zeros_like(actual_codes, dtype=bool)
    mismatch_rows = np.unique(np.argwhere(actual_codes != expected_codes)[:, 0])
    for row in mismatch_rows:
        direct, propagated = _adjudicate_banded_foem_row(
            source[row], expected_codes[row], actual_codes[row]
        )
        mismatches = set(np.flatnonzero(actual_codes[row] != expected_codes[row]))
        assert mismatches == direct | propagated
        accepted[row, list(mismatches)] = True
    np.testing.assert_allclose(
        output[~accepted], expected[0][~accepted], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(actual_codes[~accepted], expected_codes[~accepted])


def test_foem_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        foem_quantize_weight_mlx(mx.zeros((0, 32)), mx.eye(32), group_size=32)
    with pytest.raises(ValueError, match="group_size"):
        foem_quantize_weight_mlx(mx.zeros((2, 33)), mx.eye(33), group_size=32)
    with pytest.raises(ValueError, match="diagonal"):
        foem_quantize_weight_mlx(mx.zeros((2, 32)), mx.zeros((32, 32)), group_size=32)
    with pytest.raises(TypeError, match="return_residuals"):
        foem_quantize_weight_mlx(
            mx.zeros((2, 32)), mx.eye(32), group_size=32, return_residuals=1
        )
