# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ packing reference: ModelCloud.ai, Apache-2.0, gptqmodel/nn_modules/qlinear/gemv_fast_awq.py
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Independent code and output checks for both AWQ GEMV checkpoint layouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tests.qwen38_27b_shapes import QWEN38_27B_PROJECTIONS  # noqa: E402


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("variant,group_size", [
    ("gemv", 64), ("gemv", 128), ("gemv", -1),
    ("fast", 16), ("fast", 32), ("fast", 64), ("fast", 128), ("fast", -1),
    ("llm", 16), ("llm", 32), ("llm", 64), ("llm", 128), ("llm", -1),
])
def test_awq_gemv_packed_transfer_matches_codes_and_torch(variant, group_size, dtype, monkeypatch):
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
    from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear, pack_intweight
    from gptqmodel.nn_modules.qlinear.mlx import (AwqGemvFastMlxQuantLinear,
                                                  AwqGemvMlxQuantLinear, LLMAwqMlxQuantLinear)
    from gptqmodel.nn_modules.qlinear.mlx_awq import MlxAWQLinear
    from gptqmodel.utils import mlx as mlx_utils

    input_dims, output_dims = 256, 64
    group = input_dims if group_size == -1 else group_size
    groups = input_dims // group
    rng = np.random.default_rng(2100 + group_size + {"gemv": 0, "fast": 10, "llm": 20}[variant])
    source_class = ({"gemv": AwqGEMVLinear, "fast": AwqGEMVFastLinear,
                     "llm": LLMAwqLinear}[variant] if not (variant == "gemv" and group_size == -1)
                    else AwqGemvMlxQuantLinear)
    holder_class = {"gemv": AwqGemvMlxQuantLinear, "fast": AwqGemvFastMlxQuantLinear,
                    "llm": LLMAwqMlxQuantLinear}[variant]
    source = torch.nn.Module()
    source.linear = source_class(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16,
    )
    codes = rng.integers(0, 16, (output_dims, input_dims), dtype=np.uint32)
    zeros = rng.integers(0, 16, (output_dims, groups), dtype=np.uint32)
    scales = rng.uniform(0.002, 0.015, (output_dims, groups)).astype(np.float16)
    codes[0, 0], codes[-1, -1] = 0, 15
    zeros[0, 0], zeros[-1, -1] = 15, 0
    if variant == "gemv":
        words = np.bitwise_or.reduce(
            codes.reshape(output_dims, -1, 8) << (np.arange(8, dtype=np.uint32) * 4), axis=-1,
        ).astype(np.int32)
        zero_words = np.bitwise_or.reduce(
            np.pad(zeros, ((0, 0), (0, (-groups) % 8))).reshape(output_dims, -1, 8)
            << (np.arange(8, dtype=np.uint32) * 4), axis=-1,
        ).astype(np.int32)
        source.linear.qweight.copy_(torch.from_numpy(words))
        source.linear.qzeros[:, :zero_words.shape[1]].copy_(torch.from_numpy(zero_words))
        source.linear.scales[:, :groups].copy_(torch.from_numpy(scales))
    else:
        source.linear.qweight.copy_(pack_intweight(torch.from_numpy(codes.astype(np.int32)), 4, 64))
        source.linear.scales[:groups].copy_(torch.from_numpy(scales.T))
        scaled_zeros = -(zeros.astype(np.float32) * scales.astype(np.float32)).astype(np.float16)
        source.linear._runtime_zeros()[:groups].copy_(torch.from_numpy(scaled_zeros.T))
    assert holder_class.source_compatible(source.linear)

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(input_dims, output_dims, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, SimpleNamespace(from_dict=lambda _: None)))
    model, _ = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    runtime_linear = model.linear.linear if isinstance(model.linear, MlxAWQLinear) else model.linear
    packed = np.asarray(runtime_linear.weight).astype(np.uint32)
    unpacked = ((packed[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(output_dims, input_dims)
    np.testing.assert_array_equal(unpacked, codes)

    inputs = rng.normal(0, 0.2, (2, 3, input_dims)).astype(np.float32)
    mlx_inputs = mx.array(inputs).astype(dtype)
    actual = model(mlx_inputs)
    mx.eval(actual)
    assert actual.dtype == dtype
    decoded = ((codes.astype(np.float32) - np.repeat(zeros, group, axis=1))
               * np.repeat(scales, group, axis=1)).astype(np.float32)
    torch_inputs = torch.from_numpy(np.asarray(mlx_inputs.astype(mx.float32))).float()
    expected = (torch_inputs @ torch.from_numpy(decoded).T).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("dtype", (mx.float16, mx.bfloat16), ids=("fp16", "bf16"))
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_awq_qwen38_outputs_preserve_dtype_and_match_torch(
    name, out_features, in_features, dtype, record_property,
):
    bits, group_size = 4, 128
    rng = np.random.default_rng(sum(name.encode()) + out_features + in_features + 41)
    codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint8)
    zeros = rng.integers(0, 16, (in_features // group_size, out_features), dtype=np.uint8)
    scales = rng.uniform(0.0005, 0.002, zeros.shape).astype(np.float16)
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    shifts = np.arange(8, dtype=np.uint32) * 4
    qweight = np.bitwise_or.reduce(
        codes.reshape(in_features, -1, 8)[:, :, order].astype(np.uint32) << shifts, axis=-1,
    )
    qzeros = np.bitwise_or.reduce(
        zeros.reshape(-1, out_features // 8, 8)[:, :, order].astype(np.uint32) << shifts, axis=-1,
    )

    from gptqmodel.utils.mlx_packing import repack_awq_4bit
    from gptqmodel.nn_modules.qlinear.mlx_awq import MlxAWQLinear

    packed, mlx_scales, biases = repack_awq_4bit(qweight, qzeros, scales, in_features, out_features)
    linear = nn.QuantizedLinear(
        in_features, out_features, bias=False, group_size=group_size, bits=bits,
    )
    linear.weight = mx.array(packed)
    linear.scales = mx.array(mlx_scales)
    linear.biases = mx.array(biases)
    layer = MlxAWQLinear(linear)

    x_source = rng.normal(0, 0.15, (3, in_features)).astype(np.float32)
    x = mx.array(x_source).astype(dtype)
    internal = linear(x)
    mx.eval(internal)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype

    scale_expanded = np.repeat(scales, group_size, axis=0).astype(np.float64)
    zeros_expanded = np.repeat(zeros, group_size, axis=0).astype(np.float64)
    torch_weight = (torch.from_numpy(codes.astype(np.float64)) - torch.from_numpy(zeros_expanded)) * torch.from_numpy(scale_expanded)
    torch_input = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    raw_oracle = (torch_input @ torch_weight).numpy()
    rounded_oracle = (torch_input @ torch_weight).to(
        torch.float16 if dtype == mx.float16 else torch.bfloat16,
    ).float().numpy()
    internal_values = np.asarray(internal.astype(mx.float32))
    visible = np.asarray(actual.astype(mx.float32))
    record_property("max_abs_internal_fp32", float(np.max(np.abs(internal_values - raw_oracle))))
    record_property("max_abs_vs_rounded_torch", float(np.max(np.abs(visible - rounded_oracle))))
    record_property("max_abs_vs_fp64_torch", float(np.max(np.abs(visible - raw_oracle))))
    np.testing.assert_allclose(internal_values, raw_oracle, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(visible, rounded_oracle, rtol=2e-3, atol=2e-3)
