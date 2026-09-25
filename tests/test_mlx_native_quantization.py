# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Apple silicon coverage for native MLX GPTQ and AWQ quantization."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")

# Keep this optional-backend test runnable in a small MLX-only environment.
_source = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_native.py"
_spec = importlib.util.spec_from_file_location("gptqmodel_mlx_native_test", _source)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


def _reference_gptq_weight(weight, hinv, bits, group_size):
    from mlx_lm.quant.gptq import quantize

    rows, columns = weight.shape
    corrected = weight.astype(mx.float32)
    scales, biases = [], []
    for start in range(0, columns, group_size):
        end = start + group_size
        errors = mx.zeros((rows, group_size), dtype=mx.float32)
        _, scale, bias = mx.quantize(
            corrected[:, start:end],
            bits=bits,
            group_size=group_size,
        )
        scales.append(scale)
        biases.append(bias)
        for column in range(start, end):
            value = corrected[:, column : column + 1]
            code = mx.clip(mx.round((value - bias) / scale), 0, 2**bits - 1)
            error = (value - (scale * code + bias)) / hinv[column, column]
            corrected[:, column:end] -= error @ hinv[column : column + 1, column:end]
            errors[:, column - start : column - start + 1] = error
            mx.eval(errors, corrected)
        corrected[:, end:] -= errors @ hinv[start:end, end:]
    scale = mx.concatenate(scales, axis=1)
    bias = mx.concatenate(biases, axis=1)
    packed = quantize(corrected, bits, scale, bias)
    mx.eval(packed, scale, bias)
    return packed, scale, bias


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fused_gptq_matches_mlx_reference(bits, group_size):
    mx.random.seed(1234)
    columns = 2 * group_size
    samples = mx.random.normal((2 * columns, columns))
    hessian = samples.T @ samples + 0.1 * mx.eye(columns)
    weight = mx.random.normal((17, columns))
    mx.eval(hessian, weight)
    hinv = native.inverse_hessian_mlx(hessian)

    actual = native.gptq_quantize_weight_mlx(weight, hinv, bits, group_size)
    expected = _reference_gptq_weight(weight, hinv, bits, group_size)

    # The MLX-LM reference uses the default Metal matmul for the cross-group
    # update, which can round differently from the Torch oracle.
    changed_words = mx.sum(actual[0] != expected[0]).item()
    assert changed_words <= 1
    assert bool(mx.allclose(actual[1], expected[1], rtol=1e-3, atol=2e-4).item())
    assert bool(mx.allclose(actual[2], expected[2], rtol=1e-3, atol=2e-4).item())

    torch = pytest.importorskip("torch")
    oracle = _torch_gptq_weight_oracle(
        np.asarray(weight), torch.from_numpy(np.asarray(hinv)), bits, group_size, torch
    )
    np.testing.assert_array_equal(np.asarray(actual[0]), oracle[0])
    np.testing.assert_allclose(np.asarray(actual[1]), oracle[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual[2]), oracle[2], rtol=1e-6, atol=1e-6)


def test_fused_gptq_supports_expert_weights():
    mx.random.seed(44)
    weight = mx.random.normal((3, 11, 64))
    hinv = mx.eye(64)
    expert = native.gptq_quantize_weight_mlx(weight, hinv, 4, 32)
    flat = native.gptq_quantize_weight_mlx(weight.reshape(33, 64), hinv, 4, 32)
    assert expert[0].shape == (3, 11, 8)
    assert expert[1].shape == (3, 11, 2)
    for shaped, flattened in zip(expert, flat):
        assert bool(mx.array_equal(shaped.reshape(flattened.shape), flattened).item())


def test_large_fused_gptq_stays_close_to_reference():
    mx.random.seed(9)
    width = 1024
    weight = mx.random.normal((width, width))
    hinv = mx.eye(width) + mx.triu(mx.random.normal((width, width)) * 0.001, k=1)
    mx.eval(weight, hinv)

    actual = native.gptq_quantize_weight_mlx(weight, hinv, 4, 64)
    expected = _reference_gptq_weight(weight, hinv, 4, 64)
    changed_words = mx.sum(actual[0] != expected[0]).item()
    assert changed_words / actual[0].size < 0.001

    actual_weight = mx.dequantize(*actual, bits=4, group_size=64)
    expected_weight = mx.dequantize(*expected, bits=4, group_size=64)
    rmse = mx.sqrt(mx.mean((actual_weight - expected_weight) ** 2)).item()
    assert rmse < 0.001


def test_zero_hessian_has_finite_inverse_factor():
    inverse_factor = native.inverse_hessian_mlx(mx.zeros((32, 32)))
    assert bool(mx.all(mx.isfinite(inverse_factor)).item())


@pytest.mark.parametrize("width", [64, 128, 512])
def test_hessian_accumulation_and_inverse_match_torch(width):
    torch = pytest.importorskip("torch")
    samples = np.random.default_rng(42).standard_normal(
        (4, 2 * width, width)
    ).astype(np.float32)
    samples[..., :2] = 0
    x_torch = torch.from_numpy(samples).reshape(-1, width)
    h_mlx = native._hessian_partial_mlx(mx.array(samples))
    # Double precision keeps the oracle's accumulation error below the MLX
    # float32 acceptance threshold for large calibration batches.
    x_oracle = x_torch.double()
    h_torch = x_oracle.T @ x_oracle
    h_mlx_numpy = np.asarray(h_mlx)
    h_torch_numpy = h_torch.numpy()
    relative_error = np.linalg.norm(h_mlx_numpy - h_torch_numpy) / np.linalg.norm(
        h_torch_numpy
    )
    assert np.isfinite(h_mlx_numpy).all()
    assert relative_error <= 1e-6

    # Use one Hessian for both backends so this checks factorization separately.
    h_factor_input = h_torch.float()
    diagonal = torch.diagonal(h_factor_input)
    damp = torch.maximum(
        0.01 * diagonal.mean(),
        torch.maximum(diagonal.max() * 1e-6, torch.tensor(1e-12)),
    )
    regularized = h_factor_input + torch.eye(width) * damp
    lower = torch.linalg.cholesky(regularized)
    torch_factor = torch.linalg.cholesky(torch.cholesky_inverse(lower)).T
    mlx_factor = native.inverse_hessian_mlx(mx.array(h_factor_input.numpy()))
    np.testing.assert_allclose(
        np.asarray(mlx_factor), torch_factor.numpy(), rtol=1e-6, atol=1e-6
    )


def _torch_affine_group_params(weight, bits, torch):
    """Compute MLX affine-compatible scale and bias using only Torch ops."""
    minimum = weight.amin(dim=1)
    maximum = weight.amax(dim=1).clamp_min(0)
    scale = ((maximum - minimum) / (2**bits - 1)).clamp_min(1e-7)
    use_minimum = minimum.abs() > maximum.abs()
    scale = torch.where(use_minimum, scale, -scale)
    edge = torch.where(use_minimum, minimum, maximum)
    zero_code = torch.round(edge / scale)
    at_zero = zero_code == 0
    safe_code = torch.where(at_zero, torch.ones_like(zero_code), zero_code)
    scale = torch.where(at_zero, scale, edge / safe_code)
    bias = torch.where(at_zero, torch.zeros_like(edge), edge)
    return scale[:, None], bias[:, None]


def _torch_gptq_weight_oracle(weight, factor, bits, group_size, torch):
    # Torch independently chooses affine parameters, applies the GPTQ
    # correction across columns and groups, and packs the integer codes.
    rows, columns = weight.shape
    corrected = torch.from_numpy(weight.copy())
    packed_groups, expected_scales, expected_biases = [], [], []
    values_per_word = 32 // bits
    for start in range(0, columns, group_size):
        end = start + group_size
        scales, biases = _torch_affine_group_params(
            corrected[:, start:end], bits, torch
        )
        expected_scales.append(scales)
        expected_biases.append(biases)
        errors = torch.empty((rows, group_size), dtype=torch.float32)
        codes = torch.empty((rows, group_size), dtype=torch.int64)
        for offset in range(group_size):
            column = start + offset
            value = corrected[:, column].clone()
            code = torch.round((value - biases[:, 0]) / scales[:, 0]).clamp(
                0, 2**bits - 1
            )
            error = (value - (code * scales[:, 0] + biases[:, 0])) / factor[
                column, column
            ]
            codes[:, offset] = code.to(torch.int64)
            errors[:, offset] = error
            corrected[:, column + 1 : end] -= error[:, None] * factor[
                column, column + 1 : end
            ]
        shifts = torch.arange(values_per_word, dtype=torch.int64) * bits
        packed_groups.append(
            torch.sum(codes.reshape(rows, -1, values_per_word) << shifts, dim=-1)
        )
        if end < columns:
            corrected[:, end:] -= errors @ factor[start:end, end:]

    return (
        torch.cat(packed_groups, dim=1).numpy().astype(np.uint32),
        torch.cat(expected_scales, dim=1).numpy(),
        torch.cat(expected_biases, dim=1).numpy(),
    )


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_fused_gptq_matches_torch_update_oracle(bits, group_size):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2026)
    columns = 2 * group_size
    weight = rng.standard_normal((9, columns)).astype(np.float32)
    samples = torch.from_numpy(
        rng.standard_normal((2 * columns, columns)).astype(np.float32)
    )
    hessian = samples.T @ samples + 0.1 * torch.eye(columns)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    factor = torch.linalg.cholesky(inverse).T
    actual = native.gptq_quantize_weight_mlx(
        mx.array(weight), mx.array(factor.numpy()), bits=bits, group_size=group_size
    )
    expected_packed, expected_scales, expected_biases = _torch_gptq_weight_oracle(
        weight, factor, bits, group_size, torch
    )
    np.testing.assert_array_equal(np.asarray(actual[0]), expected_packed)
    np.testing.assert_allclose(
        np.asarray(actual[1]), expected_scales,
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(actual[2]), expected_biases,
        rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parametrize("bits,group_size", [
    (3, 64), (6, 64),
    (4, -1), (4, 16), (4, 256),
])
def test_fused_gptq_rejects_unsupported_bit_and_group_sizes(bits, group_size):
    weight = mx.zeros((9, 128))
    inverse_hessian = mx.eye(128)
    with pytest.raises(ValueError):
        native.gptq_quantize_weight_mlx(
            weight, inverse_hessian, bits=bits, group_size=group_size,
        )


def test_model_uses_smaller_group_for_narrow_layers():
    class NarrowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(96, 24)
            self.second = nn.Linear(24, 96)

        def __call__(self, x):
            return self.second(self.first(x))

    model = NarrowModel()
    data = mx.random.normal((2, 8, 96))
    mx.eval(model, data)
    model, config = native.gptq_quantize_model_mlx(
        model,
        data,
        bits=4,
        group_size=64,
        fallback_group_size=64,
    )
    assert config["first"] == {"bits": 4, "group_size": 32}
    assert config["second"] is False
    assert isinstance(model.first, nn.QuantizedLinear)
    assert isinstance(model.second, nn.Linear)


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_model_entry_point_quantizes_tiny_qwen2(method, monkeypatch, tmp_path):
    import mlx_lm.quant.utils as quant_utils
    import mlx_lm.utils as mlx_utils
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=4,
    )
    model = Model(args)
    data = mx.random.randint(0, 128, (2, 32))
    mx.eval(model, data)
    saved = {}

    monkeypatch.setattr(
        mlx_utils,
        "load",
        lambda *args, **kwargs: (model, None, {"model_type": "qwen2"}),
    )
    monkeypatch.setattr(
        mlx_utils,
        "save",
        lambda path, source, model, tokenizer, config: saved.update(
            path=path,
            model=model,
            config=config,
        ),
    )
    monkeypatch.setattr(quant_utils, "load_data", lambda *args: data)

    kwargs = (
        {"n_grid": 2, "embed_group_size": 32}
        if method == "awq"
        else {
            "fallback_group_size": 32,
        }
    )
    result = native.quantize_mlx(
        str(tmp_path),
        "tiny-output",
        method=method,
        bits=4,
        group_size=32,
        num_samples=2,
        sequence_length=32,
        **kwargs,
    )

    assert result == saved["path"] == "tiny-output"
    assert saved["config"]["quantization"]["bits"] == 4
    assert isinstance(
        saved["model"].model.layers[0].self_attn.q_proj, nn.QuantizedLinear
    )
