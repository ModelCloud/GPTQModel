# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant rotation reference: Z Lab, MIT, https://github.com/z-lab/paroquant
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""ParoQuant MLX packed transfer and independent Torch accuracy checks."""

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def _packed_awq(values):
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    arranged = values.reshape(*values.shape[:-1], -1, 8)[..., order]
    return np.bitwise_or.reduce(arranged << (np.arange(8, dtype=np.uint32) * 4), axis=-1).astype(np.int32)


@pytest.mark.parametrize("group_size", [16, 32, 64, 128, -1])
@pytest.mark.parametrize("krot", [1, 8])
@pytest.mark.parametrize("desc_act", [False, True])
def test_packed_paro_rotation_matches_torch_oracle(monkeypatch, group_size, krot, desc_act):
    from gptqmodel.nn_modules.qlinear.mlx import ParoMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils import mlx as mlx_utils

    rng = np.random.default_rng(701 + group_size + krot)
    input_dims, output_dims = 128, 64
    source_group = input_dims if group_size == -1 else group_size
    groups = input_dims // source_group
    source = torch.nn.Module()
    source.linear = ParoLinear(
        bits=4, group_size=group_size, sym=True, desc_act=desc_act,
        in_features=input_dims, out_features=output_dims, bias=True,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        format=FORMAT.PAROQUANT, krot=krot,
    )
    codes = rng.integers(0, 16, (input_dims, output_dims), dtype=np.uint32)
    zeros = rng.integers(0, 16, (groups, output_dims), dtype=np.uint32)
    scales = rng.uniform(0.002, 0.015, (groups, output_dims)).astype(np.float16)
    codes[0, 0], codes[-1, -1] = 0, 15
    zeros[0, 0], zeros[-1, -1] = 15, 0
    source.linear.qweight.copy_(torch.from_numpy(_packed_awq(codes)))
    source.linear.qzeros.copy_(torch.from_numpy(_packed_awq(zeros)))
    source.linear.scales.copy_(torch.from_numpy(scales))
    source.linear.bias.copy_(torch.from_numpy(rng.uniform(-0.01, 0.01, output_dims).astype(np.float16)))
    pair_rows = np.empty((krot, groups, source_group // 2, 2), dtype=np.int16)
    for stage in range(krot):
        for group in range(groups):
            pair_rows[stage, group] = rng.permutation(source_group).reshape(-1, 2)
    angles = rng.uniform(-0.035, 0.035, (krot, input_dims // 2)).astype(np.float16)
    channel_scales = rng.uniform(0.9, 1.1, (1, input_dims)).astype(np.float16)
    source.linear.pairs.copy_(torch.from_numpy(pair_rows.reshape(krot, input_dims)))
    source.linear.theta.copy_(torch.from_numpy(angles))
    source.linear.channel_scales.copy_(torch.from_numpy(channel_scales))
    assert ParoMlxQuantLinear.source_compatible(source.linear)

    class Tiny(nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = nn.Linear(input_dims, output_dims, bias=True)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (Tiny, SimpleNamespace(from_dict=lambda _: None)))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, MlxParoLinear)
    assert config["_gptqmodel_custom_mlx_runtime"]
    packed = np.array(model.linear.linear.weight)
    np.testing.assert_array_equal(
        (packed[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4) & 15).reshape(output_dims, input_dims),
        codes.T,
    )

    inputs = rng.normal(0, 0.2, (2, 3, input_dims)).astype(np.float16)
    actual = model(mx.array(inputs))
    mx.eval(actual)
    assert actual.dtype == mx.float16
    rotated = torch.from_numpy(inputs).double()
    rotated = rotated * torch.from_numpy(channel_scales).double()
    for stage in range(krot):
        next_rotated = rotated.clone()
        for group in range(groups):
            for pair in range(source_group // 2):
                i, j = map(int, pair_rows[stage, group, pair])
                i += group * source_group
                j += group * source_group
                angle = float(angles[stage, group * source_group // 2 + pair])
                c, s = np.cos(angle), np.sin(angle)
                next_rotated[..., i] = rotated[..., i] * c + rotated[..., j] * s
                next_rotated[..., j] = -rotated[..., i] * s + rotated[..., j] * c
        rotated = next_rotated
    expanded_zeros = np.repeat(zeros.astype(np.float64), source_group, axis=0)
    expanded_scales = np.repeat(scales.astype(np.float64), source_group, axis=0)
    weights = torch.from_numpy((codes.astype(np.float64) - expanded_zeros) * expanded_scales)
    expected = rotated @ weights + source.linear.bias.double()
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=0.002, atol=0.002)


def test_invalid_paro_matching_is_rejected():
    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, ParoMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
    from gptqmodel.quantization.config import FORMAT

    layer = ParoLinear(
        bits=4, group_size=32, sym=True, desc_act=False, in_features=128, out_features=64,
        register_buffers=True, dtype=torch.float16, format=FORMAT.PAROQUANT, krot=1,
    )
    assert not AwqMlxQuantLinear.source_compatible(layer)
    layer.pairs[0, 1] = layer.pairs[0, 0]
    assert not ParoMlxQuantLinear.source_compatible(layer)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_rotated_paro_preserves_activation_dtype(dtype):
    from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear

    linear = nn.QuantizedLinear(128, 64, bias=False, group_size=128, bits=4)
    linear.load_weights([
        ("weight", mx.zeros((64, 16), dtype=mx.uint32)),
        ("scales", mx.full((64, 1), 0.01, dtype=mx.float16)),
        ("biases", mx.full((64, 1), -0.08, dtype=mx.float32)),
    ])
    layer = MlxParoLinear(
        linear, np.arange(128, dtype=np.int16).reshape(1, -1),
        np.full((1, 64), 0.01, dtype=np.float16),
        np.full((1, 128), 1.01, dtype=np.float16), 128,
    )
    x = mx.ones((2, 128), dtype=dtype)
    actual, internal = layer(x), layer._forward_unrounded(x)
    mx.eval(actual, internal)
    assert internal.dtype == mx.float32
    assert actual.dtype == dtype
    assert np.isfinite(np.asarray(actual.astype(mx.float32))).all()


@pytest.mark.parametrize("name,output_dims,input_dims", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("krot", [1, 8])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_qwen38_paro_fused_rotation_matches_torch(name, output_dims, input_dims, krot, dtype):
    from gptqmodel.nn_modules.qlinear.mlx_paro import MlxParoLinear

    rng = np.random.default_rng(380027)
    group_size = 128
    pairs = np.empty((krot, input_dims), dtype=np.int16)
    for stage in range(krot):
        for group in range(input_dims // group_size):
            pairs[stage, group * group_size:(group + 1) * group_size] = rng.permutation(group_size)
    angles = rng.uniform(-0.025, 0.025, (krot, input_dims // 2)).astype(np.float16)
    channel_scales = rng.uniform(0.95, 1.05, (1, input_dims)).astype(np.float16)

    phase = np.arange(output_dims, dtype=np.uint8)[:, None] % 16
    index = np.arange(input_dims, dtype=np.uint8)[None, :] % 16
    codes = ((index + phase) % 16).astype(np.uint8)
    lanes = codes.reshape(output_dims, input_dims // 8, 8)
    packed = np.zeros((output_dims, input_dims // 8), dtype=np.uint32)
    for lane in range(8):
        packed |= lanes[:, :, lane].astype(np.uint32) << (4 * lane)
    scale = np.float16(0.01)
    linear = nn.QuantizedLinear(input_dims, output_dims, bias=False,
                                group_size=128, bits=4)
    linear.load_weights([
        ("weight", mx.array(packed)),
        ("scales", mx.full((output_dims, input_dims // 128), scale, dtype=mx.float16)),
        ("biases", mx.full((output_dims, input_dims // 128), -8 * float(scale), dtype=mx.float32)),
    ])
    layer = MlxParoLinear(linear, pairs, angles, channel_scales, group_size)
    torch.manual_seed(380027)
    x = (torch.randn(3, input_dims) * 0.05).half()
    x = x.to(torch.bfloat16) if dtype == mx.bfloat16 else x
    mlx_x = mx.array(x.float().numpy()).astype(dtype)
    actual = layer(mlx_x)
    internal = layer._forward_unrounded(mlx_x)
    mx.eval(actual, internal)
    assert actual.dtype == dtype

    # Separate Torch oracle: pairwise rotations in float64, followed by a
    # 16-phase dense matmul whose rows are tiled across the projection output.
    rotated = x.double() * torch.from_numpy(channel_scales).double()
    offsets = torch.arange(input_dims // group_size).repeat_interleave(group_size // 2) * group_size
    for stage in range(krot):
        pair_row = torch.from_numpy(pairs[stage].reshape(-1, 2).astype(np.int64))
        first, second = pair_row[:, 0] + offsets, pair_row[:, 1] + offsets
        cosine = torch.cos(torch.from_numpy(angles[stage]).double())
        sine = torch.sin(torch.from_numpy(angles[stage]).double())
        left, right = rotated[:, first], rotated[:, second]
        next_rotated = torch.empty_like(rotated)
        next_rotated[:, first] = left * cosine + right * sine
        next_rotated[:, second] = -left * sine + right * cosine
        rotated = next_rotated
    pattern = (torch.arange(input_dims)[None, :] + torch.arange(16)[:, None]).remainder(16).double() - 8
    expected_phases = rotated @ (pattern * float(scale)).T
    expected = expected_phases[:, np.arange(output_dims) % 16]
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), expected.numpy(),
                               rtol=0.002, atol=0.002, err_msg=f"{name}, krot={krot}")
    max_abs_error = np.max(np.abs(np.asarray(internal).astype(np.float64) - expected.numpy()))
    assert max_abs_error < 2e-4, f"{name}, krot={krot}: max_abs_error={max_abs_error}"
    del layer, linear, packed, codes, actual
    mx.clear_cache()
    gc.collect()
