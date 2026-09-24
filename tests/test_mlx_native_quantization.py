# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Apple silicon coverage for native MLX GPTQ and AWQ quantization."""

import importlib.util
import sys
from pathlib import Path

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

    assert bool(mx.array_equal(actual[0], expected[0]).item())
    assert bool(mx.allclose(actual[1], expected[1], rtol=1e-3, atol=2e-4).item())
    assert bool(mx.allclose(actual[2], expected[2], rtol=1e-3, atol=2e-4).item())


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
