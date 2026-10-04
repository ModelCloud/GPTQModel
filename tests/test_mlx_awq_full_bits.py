# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# AWQ reference: MIT Han Lab, MIT License, https://github.com/mit-han-lab/llm-awq
# Qwen projection shapes: Qwen Team, Apache-2.0, https://huggingface.co/Qwen
"""Full-bit AWQ accuracy coverage for native MLX inference."""

import gc

import numpy as np
import pytest
import torch

from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from gptqmodel.nn_modules.qlinear.mlx_awq import MlxAWQGroup16Linear, MlxAWQLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear  # noqa: E402
from gptqmodel.utils.mlx_packing import repack_awq  # noqa: E402


BITS = (2, 3, 4, 5, 6, 7, 8)
GROUP_SIZES = (-1, 16, 32, 64, 128)
DTYPES = (mx.float16, mx.bfloat16)


def _pack_stream(codes, bits):
    """Independent little-endian stream packer used as the test oracle."""
    rows, count = codes.shape
    words = np.zeros((rows, (count * bits + 31) // 32), dtype=np.uint32)
    for index in range(count):
        bit = index * bits
        word, shift = divmod(bit, 32)
        value = codes[:, index].astype(np.uint64)
        words[:, word] |= ((value << shift) & 0xFFFFFFFF).astype(np.uint32)
        if shift + bits > 32:
            words[:, word + 1] |= (value >> (32 - shift)).astype(np.uint32)
    return words


def _pack_awq_source(codes, bits):
    if bits == 4:
        codes = codes.reshape(codes.shape[0], -1, 8)[..., [0, 2, 4, 6, 1, 3, 5, 7]].reshape(codes.shape)
    return _pack_stream(codes, bits)


def _load_layer(weight, scales, biases, bits, group_size):
    output_dims, packed_dims = weight.shape
    input_dims = packed_dims * 32 // bits
    if group_size == 16:
        layer = MlxAWQGroup16Linear(input_dims, output_dims, bits)
        layer.weight = mx.array(weight)
        layer.scales_even = mx.array(scales[:, ::2]).astype(mx.float32)
        layer.scales_odd = mx.array(scales[:, 1::2]).astype(mx.float32)
        layer.biases_even = mx.array(biases[:, ::2])
        layer.biases_odd = mx.array(biases[:, 1::2])
        return layer
    native = nn.QuantizedLinear(
        input_dims, output_dims, bias=False,
        group_size=max(32, group_size), bits=bits,
    )
    native.load_weights([
        ("weight", mx.array(weight)),
        ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    return MlxAWQLinear(native)


@pytest.mark.parametrize("bits", BITS)
def test_awq_torch_packer_round_trips_every_bit(bits):
    """Cover the standard AWQ processor's dense-to-checkpoint packing path."""
    rng = np.random.default_rng(2700 + bits)
    in_features, out_features, group_size = 128, 64, 32
    groups = in_features // group_size
    codes = rng.integers(0, 1 << bits, (in_features, out_features), dtype=np.int32)
    zeros = rng.integers(0, 1 << bits, (groups, out_features), dtype=np.int32)
    scales = rng.uniform(0.01, 0.05, (groups, out_features)).astype(np.float32)
    expanded_zeros = np.repeat(zeros, group_size, axis=0)
    expanded_scales = np.repeat(scales, group_size, axis=0)
    dense = ((codes - expanded_zeros) * expanded_scales).T.astype(np.float32)
    source = torch.nn.Linear(in_features, out_features, bias=False)
    source.weight.data.copy_(torch.from_numpy(dense))
    packed = AwqTorchLinear(
        bits=bits, group_size=group_size, sym=False, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
        register_buffers=False, dtype=torch.float16,
    )
    packed.pack(
        source,
        torch.from_numpy(scales.T),
        torch.from_numpy(zeros.T.astype(np.float32)),
    )
    expected_weight = torch.from_numpy(dense.T)
    actual_weight = packed.forward(torch.eye(in_features, dtype=torch.float16)).float()
    np.testing.assert_allclose(
        actual_weight.numpy(), expected_weight.numpy(),
        rtol=0.005, atol=0.005,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sym", (False, True))
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("bits", BITS)
def test_awq_every_bit_group_symmetry_and_dtype(bits, group_size, sym, dtype):
    """Check every advertised AWQ combination against independent Torch math."""
    rng = np.random.default_rng(380027 + bits * 100 + (group_size % 997) + int(sym))
    in_features, out_features = 128, 64
    effective_group = in_features if group_size == -1 else group_size
    groups = in_features // effective_group
    codes = rng.integers(0, 1 << bits, (in_features, out_features), dtype=np.uint32)
    if sym:
        zeros = np.full((groups, out_features), 1 << (bits - 1), dtype=np.uint32)
    else:
        zeros = rng.integers(0, 1 << bits, (groups, out_features), dtype=np.uint32)
    scales = rng.uniform(0.0002, 0.001, zeros.shape).astype(np.float16)
    qweight = _pack_awq_source(codes, bits)
    qzeros = _pack_awq_source(zeros, bits)
    weight, mlx_scales, biases = repack_awq(
        qweight, qzeros, scales, in_features, out_features, bits,
    )
    target_bits = 8 if bits == 7 else bits
    target_group = max(32, min(effective_group, 128))
    if effective_group > target_group:
        repeats = effective_group // target_group
        mlx_scales = mlx_scales.repeat(repeats, axis=1)
        biases = biases.repeat(repeats, axis=1)
    np.testing.assert_array_equal(weight, _pack_stream(codes.T, target_bits))

    runtime_group = 16 if effective_group == 16 else target_group
    layer = _load_layer(weight, mlx_scales, biases, target_bits, runtime_group)
    x_numpy = rng.normal(0, 0.02, (3, in_features)).astype(np.float16)
    x = mx.array(x_numpy).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype

    visible_x = torch.from_numpy(np.asarray(x.astype(mx.float32))).double()
    expanded_zeros = torch.from_numpy(zeros.astype(np.float64)).repeat_interleave(effective_group, dim=0)
    expanded_scales = torch.from_numpy(scales.astype(np.float64)).repeat_interleave(effective_group, dim=0)
    oracle_weight = (torch.from_numpy(codes.astype(np.float64)) - expanded_zeros) * expanded_scales
    expected = visible_x @ oracle_weight
    rounded = expected.to(torch.float16 if dtype == mx.float16 else torch.bfloat16).float().numpy()
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), rounded,
        rtol=0.004, atol=0.004,
    )


def _constant_words(code, bits, count):
    block = _pack_stream(np.full((1, 32), code, dtype=np.uint32), bits)[0]
    return np.tile(block, count // 32)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("name,out_features,in_features", QWEN38_27B_PROJECTIONS)
def test_qwen38_all_awq_bits_preserve_dtype_and_accuracy(
    name, out_features, in_features, bits, dtype,
):
    """Exercise every native AWQ width on each Qwen3.8-27B projection shape."""
    del name
    target_bits = 8 if bits == 7 else bits
    code = (1 << bits) - 1
    zero = code - 1
    packed_row = _constant_words(code, target_bits, in_features)
    weight = np.broadcast_to(packed_row, (out_features, packed_row.size)).copy()
    scales = np.full((out_features, in_features // 128), 0.001, dtype=np.float16)
    biases = np.full(scales.shape, -zero * 0.001, dtype=np.float32)
    layer = _load_layer(weight, scales, biases, target_bits, 128)

    rng = np.random.default_rng(380027 + bits)
    x = mx.array(rng.normal(0, 0.01, (1, in_features)).astype(np.float16)).astype(dtype)
    actual = layer(x)
    mx.eval(actual)
    assert actual.dtype == dtype
    expected_value = float(np.asarray(x.astype(mx.float32)).sum()) * 0.001
    expected = torch.full((1, out_features), expected_value)
    rounded = expected.to(torch.float16 if dtype == mx.float16 else torch.bfloat16).float().numpy()
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)), rounded,
        rtol=0.004, atol=0.004,
    )
    del layer, weight
    gc.collect()
    mx.clear_cache()
