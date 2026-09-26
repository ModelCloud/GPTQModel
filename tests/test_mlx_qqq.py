# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# QQQ reference: vLLM contributors, Apache-2.0, https://github.com/vllm-project/vllm
"""QQQ packed INT8 transfer and dynamic activation arithmetic on Metal."""

import numpy as np
import pytest
import torch


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.mark.parametrize("width", [128, 5120, 17408])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qqq_metal_dynamic_quant_matches_torch_codes(width, dtype):
    from gptqmodel.nn_modules.qlinear.mlx_qqq import _dynamic_quant

    torch.manual_seed(801)
    source = torch.randn(2, 3, width, dtype=torch.float32).mul_(0.05).to(dtype)
    source[0, 0] = 0
    source[0, 1, 0] = 1
    source[0, 1, 1] = -1
    half = source.half()
    expected_scales = (half.abs().amax(dim=-1, keepdim=True) / 127).float()
    expected_codes = (half / expected_scales).round().clamp(-128, 127).to(torch.int8)
    quantized, scales = _dynamic_quant(mx.array(half.numpy()))
    mx.eval(quantized, scales)
    np.testing.assert_array_equal(np.asarray(scales), expected_scales.numpy())
    np.testing.assert_array_equal(np.asarray(quantized).astype(np.int8), expected_codes.numpy())


@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("rows", [1, 3])
def test_qqq_mlx_matches_torch_quantized_arithmetic(monkeypatch, group_size, rows):
    from gptqmodel.nn_modules.qlinear.mlx import QQQMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_qqq import MlxQQQLinear
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    from gptqmodel.utils import mlx as mlx_utils

    rng = np.random.default_rng(80 + rows + group_size)
    source = torch.nn.Module()
    source.linear = QQQTorchLinear(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=256, out_features=128, bias=True,
        register_buffers=True, dtype=torch.float16,
    )
    source.linear.B.copy_(torch.from_numpy(rng.integers(0, 2**32, source.linear.B.shape, dtype=np.uint32).view(np.int32)))
    source.linear.s_channel.copy_(torch.from_numpy(rng.uniform(0.0005, 0.002, (1, 128)).astype(np.float32)))
    if group_size != -1:
        source.linear.s_group.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, (2, 128)).astype(np.float16)))
    source.linear.bias.copy_(torch.from_numpy(rng.uniform(-0.01, 0.01, 128).astype(np.float16)))
    assert QQQMlxQuantLinear.source_compatible(source.linear)

    class Args:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(256, 128, bias=True)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, Args))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, MlxQQQLinear)
    assert config["_gptqmodel_custom_mlx_runtime"]
    weight, channel = source.linear._dequantize_weight_for_torch()
    if group_size == -1:
        assert model.linear.linear.bits == 4
        packed = np.asarray(model.linear.linear.weight).astype(np.uint32)
        decoded = ((packed[..., None] >> (4 * np.arange(8))) & 15).reshape(128, 256)
        codes = source.linear._unpack_weight_codes().numpy().T
        np.testing.assert_array_equal(decoded, codes ^ 8)
        np.testing.assert_array_equal((decoded.astype(np.int16) - 8) * 16, weight.numpy().T)
    else:
        assert model.linear.linear.bits == 8
        packed = np.asarray(model.linear.linear.weight).view(np.uint8).reshape(128, 256)
        np.testing.assert_array_equal(packed, (weight.numpy().T + 128).astype(np.uint8))

    x = rng.normal(0, 0.2, (2, rows, 256)).astype(np.float16)
    output = model(mx.array(x))
    mx.eval(output)
    x_t = torch.from_numpy(x)
    scale = (x_t.abs().amax(dim=-1, keepdim=True) / 127).float()
    quantized = (x_t / scale).round().clamp(-128, 127).to(torch.int8)
    expected = (quantized.float() @ weight.float() * scale * channel).half()
    expected += source.linear.bias
    np.testing.assert_allclose(np.array(output), expected.numpy(), rtol=0.002, atol=0.002)

    zero_output = model(mx.zeros((1, 256), dtype=mx.float16))
    mx.eval(zero_output)
    np.testing.assert_array_equal(np.asarray(zero_output), source.linear.bias.numpy()[None, :])
    empty_output = model(mx.zeros((0, 256), dtype=mx.float16))
    assert empty_output.shape == (0, 128)
