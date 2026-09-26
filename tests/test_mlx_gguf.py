# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF reference: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""GGUF compressed block to MLX affine conversion checked against Torch math."""

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


_QTYPES = ("Q1_0", "Q1_0_g128", "Q2_0", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "TQ1_0", "TQ2_0")


@pytest.mark.parametrize("qtype", _QTYPES)
def test_gguf_affine_weights_and_matmul_match_source_oracle(qtype):
    from gptqmodel.nn_modules.qlinear.gguf import _GGUF_TYPE_INFO, _dequantize_gguf_tensor_numpy
    from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFQ6KLinear
    from gptqmodel.utils.mlx_gguf_packing import repack_gguf_affine

    rng = np.random.default_rng(203 + _QTYPES.index(qtype))
    input_dims, output_dims = 256, 64
    spec = _GGUF_TYPE_INFO[qtype]
    raw = rng.integers(
        0, 256, (output_dims, input_dims // spec["block_size"], spec["type_size"]), dtype=np.uint8,
    )
    d = np.array([0.001], dtype=np.float16).view(np.uint8)
    if qtype in ("Q6_K", "TQ1_0", "TQ2_0"):
        raw[..., -2:] = d
    else:
        raw[..., :2] = d
    if qtype in ("Q4_K", "Q5_K"):
        raw[..., 2:4] = d
    source = raw.reshape(output_dims, -1)
    words, scales, biases, params = repack_gguf_affine(source, qtype, input_dims)
    expected_weights = torch.from_numpy(_dequantize_gguf_tensor_numpy(source, qtype).copy()).double()
    if qtype == "Q6_K":
        layer = MlxGGUFQ6KLinear(input_dims, output_dims)
        layer.weight = mx.array(words)
        layer.scales_even = mx.array(scales[:, ::2])
        layer.scales_odd = mx.array(scales[:, 1::2])
        layer.biases_even = mx.array(biases[:, ::2])
        layer.biases_odd = mx.array(biases[:, 1::2])
        dequantized = layer(mx.eye(input_dims, dtype=mx.float16))
        mx.eval(dequantized)
        np.testing.assert_allclose(np.array(dequantized).T, expected_weights.numpy(), rtol=0.002, atol=0.002)
    else:
        dequantized = mx.dequantize(
            mx.array(words), mx.array(scales), mx.array(biases),
            group_size=params["group_size"], bits=params["bits"],
        )
        mx.eval(dequantized)
        np.testing.assert_allclose(np.array(dequantized), expected_weights.numpy(), rtol=0.002, atol=0.002)
    x = rng.normal(0, 0.2, (2, 3, input_dims)).astype(np.float16)
    if qtype == "Q6_K":
        actual = layer(mx.array(x))
    else:
        actual = mx.quantized_matmul(
            mx.array(x), mx.array(words), mx.array(scales), mx.array(biases),
            group_size=params["group_size"], bits=params["bits"],
        )
    mx.eval(actual)
    oracle = torch.from_numpy(x).double() @ expected_weights.T
    np.testing.assert_allclose(np.array(actual), oracle.numpy(), rtol=0.002, atol=0.002)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=("fp16", "bf16"))
@pytest.mark.parametrize("bits", ["q1_0", "q1_0_g128", "q2_0", "q4_0", "q8_0", "q4_k", "q5_k", "q6_k"])
def test_gguf_holder_loads_packed_mlx_layer(monkeypatch, bits, dtype):
    import mlx.nn as nn

    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    from gptqmodel.nn_modules.qlinear.mlx import GGUFMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFLinear, MlxGGUFQ6KLinear
    from gptqmodel.utils import mlx as mlx_utils

    source = torch.nn.Module()
    source.linear = GGUFTorchLinear(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=64, bias=False,
        register_buffers=True,
    )
    raw = np.zeros(tuple(source.linear.qweight.shape), dtype=np.uint8)
    raw.reshape(64, -1, source.linear.gguf_type_size)[..., :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if bits == "q6_k":
        raw.reshape(64, -1, source.linear.gguf_type_size)[..., -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    source.linear.qweight.copy_(torch.from_numpy(raw))
    assert GGUFMlxQuantLinear.source_compatible(source.linear)

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(256, 64, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    if bits == "q6_k":
        assert isinstance(model.linear, MlxGGUFQ6KLinear)
    else:
        assert isinstance(model.linear, MlxGGUFLinear)
        assert config["_gptqmodel_custom_mlx_runtime"]
    x = mx.ones((1, 256), dtype=dtype)
    actual = model(x)
    mx.eval(actual)
    oracle = torch.from_numpy(np.ones((1, 256), dtype=np.float16)).double() @ source.linear.dequantize_weight(dtype=torch.float32).double()
    expected = oracle.to(torch.float16 if dtype == mx.float16 else torch.bfloat16).float().numpy()
    assert actual.dtype == dtype
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=("fp16", "bf16"))
@pytest.mark.parametrize("qtype", ["MXFP4", "NVFP4"])
def test_gguf_fp4_maps_to_native_mlx_float4_mode(qtype, dtype):
    from gptqmodel.nn_modules.qlinear.gguf import _dequantize_gguf_tensor_numpy
    from gptqmodel.nn_modules.qlinear.mlx_gguf import MlxGGUFLinear
    from gptqmodel.utils.mlx_gguf_packing import repack_gguf_float4

    rng = np.random.default_rng(440 + len(qtype))
    block_size, block_bytes = (32, 17) if qtype == "MXFP4" else (64, 36)
    raw = rng.integers(0, 256, (64, 256 // block_size, block_bytes), dtype=np.uint8)
    if qtype == "MXFP4":
        raw[..., 0] = 120
    else:
        raw[..., :4] = np.array([20, 22, 24, 26], dtype=np.uint8)
    source = raw.reshape(64, -1)
    words, scales, biases, params = repack_gguf_float4(source, qtype, 256)
    assert biases is None
    reference = torch.from_numpy(_dequantize_gguf_tensor_numpy(source, qtype).copy()).double()
    dequantized = mx.dequantize(mx.array(words), mx.array(scales), mode=params["mode"],
                                group_size=params["group_size"], bits=4).astype(mx.float32)
    mx.eval(dequantized)
    np.testing.assert_allclose(np.array(dequantized), reference.numpy(), rtol=0.002, atol=0.002)
    linear = nn.QuantizedLinear(
        256, 64, bias=False, group_size=params["group_size"], bits=4, mode=params["mode"],
    )
    linear.weight = mx.array(words)
    linear.scales = mx.array(scales)
    layer = MlxGGUFLinear(linear)
    x = rng.normal(0, 0.2, (2, 3, 256)).astype(np.float32)
    mlx_input = mx.array(x).astype(dtype)
    internal = linear(mlx_input)
    actual = layer(mlx_input)
    mx.eval(internal, actual)
    assert actual.dtype == dtype
    expected = (torch.from_numpy(np.asarray(mlx_input.astype(mx.float32))).double() @ reference.T)
    rounded = expected.to(torch.float16 if dtype == mx.float16 else torch.bfloat16).float()
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), rounded.numpy(), rtol=0.002, atol=0.002)
