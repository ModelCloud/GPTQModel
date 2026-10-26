# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Check GPTQ MLX output dtype and drift against independent Torch math."""

import sys

import numpy as np
import pytest

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS

if sys.platform != "darwin":
    pytest.skip("MLX Metal tests require macOS", allow_module_level=True)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
torch = pytest.importorskip("torch")

from gptqmodel.nn_modules.qlinear.mlx_gptq import MlxGPTQLinear  # noqa: E402


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_gptq_qwen38_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, dtype, record_property,
):
    bits, group_size = 4, 128
    seed = sum(name.encode()) + out_features + in_features
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 2**32, (out_features, in_features // 8), dtype=np.uint32)
    scales = rng.uniform(0.0005, 0.002, (out_features, in_features // group_size)).astype(np.float16)
    biases = rng.uniform(-0.004, 0.004, scales.shape).astype(np.float32)

    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=group_size, bits=bits,
    )
    linear.weight = mx.array(packed)
    linear.scales = mx.array(scales)
    linear.biases = mx.array(biases)
    layer = MlxGPTQLinear(linear)

    x_source = rng.normal(0, 0.15, (3, in_features)).astype(np.float32)
    x = mx.array(x_source).astype(dtype)
    internal = linear(x)
    mx.eval(internal)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype

    shifts = (np.arange(32 // bits, dtype=np.uint32) * bits).reshape(1, 1, -1)
    codes = ((packed[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(out_features, in_features)
    scales_expanded = np.repeat(scales, group_size, axis=1).astype(np.float64)
    biases_expanded = np.repeat(biases, group_size, axis=1).astype(np.float64)
    torch_weight = torch.from_numpy(codes.astype(np.float64) * scales_expanded + biases_expanded)
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    oracle = (torch_input @ torch_weight.T).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    visible = np.asarray(actual.astype(mx.float32))
    internal_values = np.asarray(internal.astype(mx.float32))
    raw_oracle = (torch_input @ torch_weight.T).numpy()
    record_property("max_abs_internal_fp32", float(np.max(np.abs(internal_values - raw_oracle))))
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - oracle))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - raw_oracle))))
    np.testing.assert_allclose(internal_values, raw_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
def test_gptq_loader_selects_dtype_preserving_linear(monkeypatch, dtype):
    import mlx.nn as mlx_nn
    from torch import nn as torch_nn

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils import mlx as mlx_utils

    in_features, out_features, group_size = 128, 64, 64
    codes = (np.arange(in_features)[:, None] + np.arange(out_features)[None, :]) % 16
    zeros = np.ones((in_features // group_size, out_features), dtype=np.uint32)
    scales = np.full(zeros.shape, 0.001, dtype=np.float16)
    shifts = np.arange(8, dtype=np.uint32) * 4
    qweight = np.bitwise_or.reduce(
        codes.T.reshape(out_features, -1, 8).astype(np.uint32) << shifts, axis=-1,
    ).T
    qzeros = np.bitwise_or.reduce(
        zeros.reshape(-1, out_features // 8, 8) << shifts, axis=-1,
    )

    source = torch_nn.Module()
    source.linear = TorchLinear(
        bits=4, group_size=group_size, sym=False, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        format=FORMAT.GPTQ_V2,
    )
    source.linear.qzero_format(2)
    source.linear.qweight.copy_(torch.from_numpy(qweight.astype(np.int32)))
    source.linear.qzeros.copy_(torch.from_numpy(qzeros.astype(np.int32)))
    source.linear.scales.copy_(torch.from_numpy(scales))

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = mlx_nn.Linear(in_features, out_features, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, MlxGPTQLinear)
    assert config["_gptqmodel_custom_mlx_runtime"]

    x = mx.array(np.full((2, in_features), 0.125, dtype=np.float32)).astype(dtype)
    output = model(x)
    mx.eval(output)
    assert output.dtype == dtype
    torch_zeros = torch.from_numpy(zeros.astype(np.float64)).repeat_interleave(group_size, dim=0)
    torch_scales = torch.from_numpy(scales.astype(np.float64)).repeat_interleave(group_size, dim=0)
    reference_weight = (torch.from_numpy(codes.astype(np.float64)) - torch_zeros) * torch_scales
    reference_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    expected = (reference_input @ reference_weight).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(np.asarray(output.astype(mx.float32)), expected, rtol=2e-3, atol=2e-3)
