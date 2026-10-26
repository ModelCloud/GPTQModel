# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPTQ method: Elias Frantar et al., https://arxiv.org/abs/2210.17323
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen

"""Model-scale MLX GPTQ checks with a diagonal inverse Hessian Torch oracle."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("Metal kernels require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

_source = Path(__file__).resolve().parents[1] / "gptqmodel/quantization/mlx_native.py"
_spec = importlib.util.spec_from_file_location("gptqmodel_mlx_native_shape_audit", _source)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_qwen38_27b_gptq_full_projection_diagonal_hessian(
    name, out_features, in_features, dtype, group_size, bits,
):
    """Compare every supported GPTQ code and parameter for FP16/BF16 weights."""
    del name
    rng = np.random.default_rng(4500 + out_features + in_features)
    source = rng.normal(0, 0.5, (out_features, in_features)).astype(np.float32)
    weight = mx.array(source).astype(dtype)
    mx.eval(weight)
    oracle_weight = torch.from_numpy(np.asarray(weight.astype(mx.float32))).reshape(
        out_features, -1, group_size,
    )
    minimum = oracle_weight.amin(dim=-1)
    maximum = oracle_weight.amax(dim=-1).clamp_min(0)
    scales = ((maximum - minimum) / (2**bits - 1)).clamp_min(1e-7)
    use_minimum = minimum.abs() > maximum.abs()
    scales = torch.where(use_minimum, scales, -scales)
    edge = torch.where(use_minimum, minimum, maximum)
    zero_code = torch.round(edge / scales)
    at_zero = zero_code == 0
    scales = torch.where(
        at_zero, scales, edge / torch.where(at_zero, torch.ones_like(zero_code), zero_code),
    )
    biases = torch.where(at_zero, torch.zeros_like(edge), edge)
    codes = torch.round(
        (oracle_weight - biases[..., None]) / scales[..., None]
    ).clamp(0, 2**bits - 1).to(torch.int64)
    values_per_word = 32 // bits
    shifts = (torch.arange(values_per_word, dtype=torch.int64) * bits).reshape(
        1, 1, 1, values_per_word
    )
    expected_packed = (
        (
            codes.reshape(
                out_features, -1, group_size // values_per_word, values_per_word
            )
            << shifts
        )
        .sum(dim=-1).reshape(out_features, -1).numpy().astype(np.uint32)
    )
    actual = native.gptq_quantize_weight_mlx(
        weight, mx.eye(in_features), bits=bits, group_size=group_size,
    )
    np.testing.assert_allclose(np.asarray(actual[1]), scales.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual[2]), biases.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual[0]), expected_packed)


@pytest.mark.parametrize("in_features", sorted({shape[2] for shape in QWEN38_27B_PROJECTIONS}))
def test_qwen38_27b_hessian_accumulation_width(in_features):
    """Check each distinct model input width against float64 Torch accumulation."""
    rng = np.random.default_rng(5500 + in_features)
    activations = rng.normal(0, 0.2, (16, in_features)).astype(np.float32)
    actual = np.asarray(native._hessian_partial_mlx(mx.array(activations)))
    source = torch.from_numpy(activations).double()
    expected = (source.T @ source).numpy()
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert np.isfinite(actual).all()
    assert error <= 1e-6
