# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from types import SimpleNamespace

import numpy as np
import pytest


mlx = pytest.importorskip("mlx.core")

from gptqmodel.utils.mlx_packing import repack_awq_4bit, repack_gptq, repack_gptq_4bit  # noqa: E402


def _oracle_words(codes, bits, planar=False):
    """Pack source or MLX words one code at a time, independent of the converter."""
    rows, count = codes.shape
    words = np.zeros((rows, count * bits // 32), dtype=np.uint32)
    for row in range(rows):
        for col in range(count):
            value = int(codes[row, col])
            if planar:
                planes = {3: ((2, 0), (1, 2)), 5: ((4, 0), (1, 4)),
                          6: ((4, 0), (2, 4)), 7: ((4, 0), (2, 4), (1, 6))}[bits]
                plane_start = (col // 32) * bits
                for width, offset in planes:
                    word = plane_start + (col % 32) // (32 // width)
                    shift = ((col % 32) % (32 // width)) * width
                    words[row, word] |= np.uint32(((value >> offset) & ((1 << width) - 1)) << shift)
                    plane_start += width
            else:
                bit = col * bits
                words[row, bit // 32] |= np.uint32((value << (bit % 32)) & 0xffffffff)
                if bit % 32 + bits > 32:
                    words[row, bit // 32 + 1] |= np.uint32(value >> (32 - bit % 32))
    return words


@pytest.mark.parametrize("bits,planar", [(2, False), (3, False), (3, True), (4, False),
                                          (5, True), (6, True), (7, True), (8, False)])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256, 512, 1024])
def test_gptq_all_mlx_affine_bits_and_groups_match_torch_oracle(bits, planar, group_size):
    import torch

    rng = np.random.default_rng(400 + bits * 10 + group_size)
    in_features, out_features = max(128, group_size), 64
    codes = rng.integers(0, 1 << bits, (in_features, out_features), dtype=np.uint32)
    zeros = rng.integers(0, 1 << bits, (in_features // group_size, out_features), dtype=np.uint32)
    codes[0, 0], codes[-1, -1] = 0, (1 << bits) - 1
    zeros[0, 0], zeros[-1, -1] = (1 << bits) - 1, 0
    scales = rng.uniform(0.0001, 0.001, zeros.shape).astype(np.float16)
    qweight = _oracle_words(codes.T, bits, planar).T.copy()
    qzeros = _oracle_words(zeros, bits, planar)
    weight, mlx_scales, biases = repack_gptq(
        qweight, qzeros, scales, in_features, out_features, bits, planar,
    )
    target_bits = 8 if bits == 7 else bits
    target_group = min(group_size, 128)
    np.testing.assert_array_equal(weight, _oracle_words(codes.T, target_bits))
    np.testing.assert_array_equal(mlx_scales, scales.T)
    if group_size > target_group:
        repeats = group_size // target_group
        mlx_scales = mlx_scales.repeat(repeats, axis=1)
        biases = biases.repeat(repeats, axis=1)

    expanded_zeros = torch.from_numpy(zeros.astype(np.int64)).repeat_interleave(group_size, dim=0)
    expanded_scales = torch.from_numpy(scales).double().repeat_interleave(group_size, dim=0)
    oracle_weight = ((torch.from_numpy(codes.astype(np.int64)) - expanded_zeros).double() * expanded_scales)
    dequantized = mlx.dequantize(
        mlx.array(weight), mlx.array(mlx_scales).astype(mlx.float32), mlx.array(biases), target_group, target_bits,
    )
    mlx.eval(dequantized)
    np.testing.assert_allclose(np.array(dequantized), oracle_weight.T.numpy(), rtol=0.002, atol=0.002)

    x_numpy = rng.normal(0, 0.2, (2, 3, in_features)).astype(np.float16)
    output = mlx.quantized_matmul(
        mlx.array(x_numpy), mlx.array(weight), mlx.array(mlx_scales).astype(mlx.float32), mlx.array(biases),
        group_size=target_group, bits=target_bits,
    )
    mlx.eval(output)
    expected = torch.from_numpy(x_numpy).double() @ oracle_weight
    np.testing.assert_allclose(np.array(output), expected.numpy(), rtol=0.002, atol=0.002)


@pytest.mark.parametrize("format", ["gptq", "awq"])
@pytest.mark.parametrize("group_size", [32, 64, 128, -1])
def test_packed_4bit_matches_source_codes(format, group_size):
    import torch

    rng = np.random.default_rng(17)
    in_features, out_features = (256 if group_size == -1 else 128), 64
    effective_group = in_features if group_size == -1 else group_size
    codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint32)
    zeros = rng.integers(0, 16, (in_features // effective_group, out_features), dtype=np.uint32)
    codes[0, 0], codes[-1, -1] = 0, 15
    zeros[0, 0], zeros[-1, -1] = 15, 0
    scales = rng.uniform(0.01, 0.1, zeros.shape).astype(np.float16)
    shifts = np.arange(8, dtype=np.uint32) * 4

    if format == "gptq":
        qweight = np.bitwise_or.reduce(codes.reshape(-1, 8, out_features) << shifts[None, :, None], axis=1)
        qzeros = np.bitwise_or.reduce(zeros.reshape(-1, out_features // 8, 8) << shifts, axis=-1)
        repack = repack_gptq_4bit
    else:
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        qweight = np.bitwise_or.reduce(codes.reshape(in_features, -1, 8)[:, :, order] << shifts, axis=-1)
        qzeros = np.bitwise_or.reduce(zeros.reshape(-1, out_features // 8, 8)[:, :, order] << shifts, axis=-1)
        repack = repack_awq_4bit

    weight, mlx_scales, biases = repack(qweight, qzeros, scales, in_features, out_features)
    target_group = min(effective_group, 128)
    if effective_group > target_group:
        mlx_scales = mlx_scales.repeat(effective_group // target_group, axis=1)
        biases = biases.repeat(effective_group // target_group, axis=1)
    assert weight.dtype == np.uint32
    assert weight.shape == (out_features, in_features // 8)
    unpacked = ((weight[:, :, None] >> shifts) & 15).reshape(out_features, in_features)
    np.testing.assert_array_equal(unpacked, codes.T)

    # Independent Torch arithmetic is the numerical oracle for MLX inference.
    torch_codes = torch.from_numpy(codes.astype(np.int64))
    torch_zeros = torch.from_numpy(zeros.astype(np.int64)).repeat_interleave(effective_group, dim=0)
    torch_scales = torch.from_numpy(scales).double().repeat_interleave(effective_group, dim=0)
    expected = ((torch_codes - torch_zeros).double() * torch_scales).numpy()
    dequantized = mlx.dequantize(
        mlx.array(weight), mlx.array(mlx_scales).astype(mlx.float32),
        mlx.array(biases), target_group, 4
    )
    mlx.eval(dequantized)
    np.testing.assert_allclose(np.array(dequantized), expected.T, rtol=0.002, atol=0.002)

    x_numpy = rng.normal(size=(3, in_features)).astype(np.float16)
    x = mlx.array(x_numpy)
    y = mlx.quantized_matmul(x, mlx.array(weight), scales=mlx.array(mlx_scales),
                             biases=mlx.array(biases), group_size=target_group, bits=4)
    mlx.eval(y)
    oracle = (torch.from_numpy(x_numpy).double() @ torch.from_numpy(expected).double()).numpy()
    np.testing.assert_allclose(np.array(y), oracle, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("format,bits,planar", [("gptq", bits, bits in (5, 6, 7)) for bits in (2, 3, 4, 5, 6, 7, 8)]
                         + [("gptq", 3, True), ("awq", 4, False)])
@pytest.mark.parametrize("holder", ["torch", "mlx"])
@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_packed_layers_load_into_mlx_quantized_linear(monkeypatch, format, bits, planar, holder, group_size):
    import mlx.nn as mlx_nn
    import torch

    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils import mlx as mlx_utils

    source = torch.nn.Module()
    linear_class = (
        MlxQuantLinear if format == "gptq" else AwqMlxQuantLinear
    ) if holder == "mlx" else (
        TorchLinear if format == "gptq" else AwqTorchLinear
    )
    source.linear = linear_class(
        bits=bits, group_size=group_size, sym=False, desc_act=False,
        in_features=128, out_features=64, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        format=FORMAT.GPTQ_P if planar else None,
    )
    if format == "gptq" and not planar:
        source.linear.qzero_format(2)
    source.linear.qweight.fill_(0x76543210)
    source.linear.qzeros.fill_(0x11111111)
    source.linear.scales.fill_(0.001)

    class ModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TinyModel(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = mlx_nn.Linear(128, 64, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (TinyModel, ModelArgs))
    if group_size == 16:
        with mlx.stream(mlx.cpu):
            model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
        assert isinstance(model.linear, MlxGroup16Linear)
        assert config["_gptqmodel_group16_runtime"]
    else:
        model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
        assert isinstance(model.linear, mlx_nn.QuantizedLinear)
    assert config["quantization"]["bits"] == (8 if bits == 7 else bits)
    assert config["quantization"]["group_size"] == max(32, group_size)
    x = mlx.ones((1, 128), dtype=mlx.float16)
    if format == "gptq":
        expected = source.linear.dequantize_weight().float().sum(dim=0).numpy()
    else:
        expected = dequantize_gemm(source.linear.qweight, source.linear.qzeros,
                                   source.linear.scales, 4, source.linear.group_size).float().sum(dim=0).numpy()
    output = model(x)
    mlx.eval(output)
    np.testing.assert_allclose(np.array(output)[0], expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("bits,group_size,in_features", [(7, 256, 256), (4, 512, 512),
                                                          (5, 1024, 1024), (4, -1, 256),
                                                          (7, -1, 1024)])
def test_large_gptq_groups_load_with_repeated_mlx_scales(monkeypatch, bits, group_size, in_features):
    import mlx.nn as mlx_nn
    import torch

    from gptqmodel.nn_modules.qlinear.mlx import MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils import mlx as mlx_utils

    source = torch.nn.Module()
    source.linear = TorchLinear(
        bits=bits, group_size=group_size, sym=False, desc_act=False,
        in_features=in_features, out_features=64, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        format=FORMAT.GPTQ_P if bits in (5, 7) else FORMAT.GPTQ_V2,
    )
    if bits == 4:
        source.linear.qzero_format(2)
    source.linear.qweight.fill_(0x76543210)
    source.linear.qzeros.fill_(0x11111111)
    source.linear.scales.fill_(0.0002)
    assert MlxQuantLinear.source_compatible(source.linear)
    _, scales, biases, params = MlxQuantLinear.pack_source(source.linear)
    assert scales.shape == biases.shape == (64, in_features // 128)
    assert params == {"group_size": 128, "bits": 8 if bits == 7 else bits, "mode": "affine"}

    class ModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TinyModel(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = mlx_nn.Linear(in_features, 64, bias=False)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (TinyModel, ModelArgs))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, mlx_nn.QuantizedLinear)
    assert config["quantization"] == params
    x = mlx.full((1, in_features), 0.125, dtype=mlx.float16)
    actual = model(x)
    mlx.eval(actual)
    expected = source.linear.dequantize_weight().float().sum(dim=0).numpy() * 0.125
    np.testing.assert_allclose(np.array(actual)[0], expected, rtol=0.002, atol=0.002)


def test_awq_tensor_group_maps_to_mlx_128():
    import torch

    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    source = AwqTorchLinear(
        bits=4, group_size=-1, sym=False, desc_act=False,
        in_features=256, out_features=64, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
    )
    source.qweight.fill_(0x76543210)
    source.qzeros.fill_(0x11111111)
    source.scales.fill_(0.001)
    assert AwqMlxQuantLinear.source_compatible(source)
    weight, scales, biases, params = AwqMlxQuantLinear.pack_source(source)
    assert weight.shape == (64, 32)
    assert scales.shape == biases.shape == (64, 2)
    assert params == {"group_size": 128, "bits": 4, "mode": "affine"}


@pytest.mark.parametrize("format,bits,planar", [("gptq", bits, bits in (5, 6, 7))
                                               for bits in (2, 3, 4, 5, 6, 7, 8)]
                         + [("gptq", 3, True), ("awq", 4, False)])
def test_group16_mlx_linear_matches_independent_torch_oracle(monkeypatch, format, bits, planar):
    import mlx.nn as mlx_nn
    import torch

    from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import FORMAT
    from gptqmodel.utils import mlx as mlx_utils

    rng = np.random.default_rng(911 + bits + int(planar))
    in_features, out_features, group_size = 128, 64, 16
    codes = rng.integers(0, 1 << bits, (in_features, out_features), dtype=np.uint32)
    zeros = rng.integers(0, 1 << bits, (in_features // group_size, out_features), dtype=np.uint32)
    codes[0, 0], codes[-1, -1] = 0, (1 << bits) - 1
    zeros[0, 0], zeros[-1, -1] = (1 << bits) - 1, 0
    scales = rng.uniform(0.0001, 0.001, zeros.shape).astype(np.float16)
    scales[0, 1] = 0
    bias = rng.uniform(-0.01, 0.01, out_features).astype(np.float16)

    source = torch.nn.Module()
    source.linear = (TorchLinear if format == "gptq" else AwqTorchLinear)(
        bits=bits, group_size=group_size, sym=False, desc_act=False,
        in_features=in_features, out_features=out_features, bias=True,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        format=FORMAT.GPTQ_P if planar else FORMAT.GPTQ_V2 if format == "gptq" else None,
    )
    if format == "gptq":
        if not planar:
            source.linear.qzero_format(2)
        qweight = _oracle_words(codes.T, bits, planar).T.copy()
        qzeros = _oracle_words(zeros, bits, planar)
    else:
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        shifts = np.arange(8, dtype=np.uint32) * 4
        qweight = np.bitwise_or.reduce(
            codes.reshape(in_features, -1, 8)[:, :, order] << shifts, axis=-1,
        )
        qzeros = np.bitwise_or.reduce(
            zeros.reshape(-1, out_features // 8, 8)[:, :, order] << shifts, axis=-1,
        )
    source.linear.qweight.copy_(torch.from_numpy(qweight.astype(np.int32)))
    source.linear.qzeros.copy_(torch.from_numpy(qzeros.astype(np.int32)))
    source.linear.scales.copy_(torch.from_numpy(scales))
    source.linear.bias.copy_(torch.from_numpy(bias))

    class ModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TinyModel(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.linear = mlx_nn.Linear(in_features, out_features, bias=True)

        def __call__(self, x):
            return self.linear(x)

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (TinyModel, ModelArgs))
    with mlx.stream(mlx.cpu):
        model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.linear, MlxGroup16Linear)
    assert config["_gptqmodel_group16_runtime"]
    target_bits = 8 if bits == 7 else bits
    np.testing.assert_array_equal(np.array(model.linear.weight), _oracle_words(codes.T, target_bits))

    x_numpy = rng.normal(0, 0.2, (2, 3, in_features)).astype(np.float16)
    output = model(mlx.array(x_numpy))
    mlx.eval(output)
    torch_codes = torch.from_numpy(codes.astype(np.int64))
    torch_zeros = torch.from_numpy(zeros.astype(np.int64)).repeat_interleave(group_size, dim=0)
    torch_scales = torch.from_numpy(scales).double().repeat_interleave(group_size, dim=0)
    expected_weight = (torch_codes - torch_zeros).double() * torch_scales
    expected = torch.from_numpy(x_numpy).double() @ expected_weight + torch.from_numpy(bias).double()
    np.testing.assert_allclose(np.array(output), expected.numpy(), rtol=0.002, atol=0.002)


@pytest.mark.parametrize("formats", [("gptq", "gptq"), ("awq", "awq"), ("gptq", "awq")])
@pytest.mark.parametrize("first_group_size", [16, 32])
def test_packed_mixed_layers_with_bias_match_independent_torch_oracle(monkeypatch, formats, first_group_size):
    import mlx.nn as mlx_nn
    import torch

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.nn_modules.qlinear.mlx_group16 import MlxGroup16Linear
    from gptqmodel.utils import mlx as mlx_utils

    rng = np.random.default_rng(286)
    shifts = np.arange(8, dtype=np.uint32) * 4
    awq_order = [0, 2, 4, 6, 1, 3, 5, 7]
    source = torch.nn.Module()
    oracle_weights = []
    oracle_biases = []

    for name, format, in_features, out_features, group_size in (
        ("first", formats[0], 128, 64, first_group_size),
        ("second", formats[1], 64, 64, 64),
    ):
        codes = rng.integers(0, 16, (in_features, out_features), dtype=np.uint32)
        zeros = rng.integers(0, 16, (in_features // group_size, out_features), dtype=np.uint32)
        codes[0, 0], codes[-1, -1] = 0, 15
        zeros[0, 0], zeros[-1, -1] = 15, 0
        scales = rng.uniform(0.005, 0.03, zeros.shape).astype(np.float16)
        scales[0, 1] = 0  # An exact zero scale catches bias and zero-point handling.
        bias = rng.uniform(-0.05, 0.05, out_features).astype(np.float16)
        linear_class = TorchLinear if format == "gptq" else AwqTorchLinear
        linear = linear_class(
            bits=4, group_size=group_size, sym=False, desc_act=False,
            in_features=in_features, out_features=out_features, bias=True,
            pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
        )
        if format == "gptq":
            linear.qzero_format(2)
            qweight = np.bitwise_or.reduce(
                codes.reshape(-1, 8, out_features) << shifts[None, :, None], axis=1,
            )
            qzeros = np.bitwise_or.reduce(
                zeros.reshape(-1, out_features // 8, 8) << shifts, axis=-1,
            )
        else:
            qweight = np.bitwise_or.reduce(
                codes.reshape(in_features, -1, 8)[:, :, awq_order] << shifts, axis=-1,
            )
            qzeros = np.bitwise_or.reduce(
                zeros.reshape(-1, out_features // 8, 8)[:, :, awq_order] << shifts, axis=-1,
            )
        linear.qweight.copy_(torch.from_numpy(qweight.astype(np.int32)))
        linear.qzeros.copy_(torch.from_numpy(qzeros.astype(np.int32)))
        linear.scales.copy_(torch.from_numpy(scales))
        linear.bias.copy_(torch.from_numpy(bias))
        setattr(source, name, linear)

        expanded_zeros = torch.from_numpy(zeros.astype(np.int64)).repeat_interleave(group_size, dim=0)
        expanded_scales = torch.from_numpy(scales).double().repeat_interleave(group_size, dim=0)
        oracle_weights.append((torch.from_numpy(codes.astype(np.int64)) - expanded_zeros).double() * expanded_scales)
        oracle_biases.append(torch.from_numpy(bias).double())

    class ModelArgs:
        @classmethod
        def from_dict(cls, _config):
            return cls()

    class TinyModel(mlx_nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.first = mlx_nn.Linear(128, 64, bias=True)
            self.second = mlx_nn.Linear(64, 64, bias=True)

        def __call__(self, x):
            return self.second(self.first(x))

    monkeypatch.setattr(mlx_utils, "_get_classes", lambda config: (TinyModel, ModelArgs))
    model, config = mlx_utils._packed_mlx_weights(source, {}, "lm_head")
    assert isinstance(model.first, MlxGroup16Linear if first_group_size == 16 else mlx_nn.QuantizedLinear)
    assert isinstance(model.second, mlx_nn.QuantizedLinear)
    assert config["quantization"]["group_size"] == 32
    assert config["quantization"]["second"]["group_size"] == 64

    x_numpy = rng.normal(0, 0.2, (2, 3, 128)).astype(np.float16)
    first_actual = model.first(mlx.array(x_numpy))
    actual = model(mlx.array(x_numpy))
    mlx.eval(first_actual, actual)
    expected = torch.from_numpy(x_numpy).double()
    expected = expected @ oracle_weights[0] + oracle_biases[0]
    first_error = np.max(np.abs(np.array(first_actual) - expected.numpy()))
    assert first_error <= 0.002
    np.testing.assert_allclose(np.array(first_actual), expected.numpy(), rtol=0.002, atol=0.002)
    expected = expected @ oracle_weights[1] + oracle_biases[1]
    final_error = np.max(np.abs(np.array(actual) - expected.numpy()))
    assert final_error <= 0.002
    np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=0.002, atol=0.002)


def test_packed_gptq_rejects_reordered_group_indices_and_legacy_zeros():
    import torch

    from gptqmodel.nn_modules.qlinear.mlx import MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    source = TorchLinear(
        bits=4, group_size=64, sym=False, desc_act=False,
        in_features=128, out_features=64, bias=False,
        pack_dtype=torch.int32, register_buffers=True, dtype=torch.float16,
    )
    source.qzero_format(2)
    assert MlxQuantLinear.source_compatible(source)
    source.g_idx[0] = 1
    assert not MlxQuantLinear.source_compatible(source)
    source.g_idx[0] = 0
    source.qzero_format(1)
    assert not MlxQuantLinear.source_compatible(source)


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal requires macOS")
@pytest.mark.parametrize("format", ["gptq", "awq"])
def test_mlx_quant_linear_registry_validates_capabilities(format):
    import torch

    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.utils.importer import select_quant_linear, validate_quant_linear

    linear_class, method, checkpoint_format = (
        (MlxQuantLinear, METHOD.GPTQ, FORMAT.GPTQ_V2) if format == "gptq"
        else (AwqMlxQuantLinear, METHOD.AWQ, FORMAT.GEMM)
    )
    contract = dict(
        bits=4, group_size=64, desc_act=False, sym=False,
        pack_dtype=torch.int32, dtype=torch.float16,
        in_features=128, out_features=64, device=DEVICE.MPS,
    )
    assert select_quant_linear(
        **{key: contract[key] for key in ("bits", "group_size", "desc_act", "sym", "pack_dtype", "dtype", "device")},
        backend=BACKEND.MLX, format=checkpoint_format, quant_method=method,
    ) is linear_class
    assert validate_quant_linear(linear_class, **contract)[0]
    assert validate_quant_linear(linear_class, **(contract | {"group_size": 16}))[0]
    assert validate_quant_linear(linear_class, **(contract | {"group_size": -1}))[0]
    assert not validate_quant_linear(linear_class, **(contract | {"group_size": -1, "in_features": 96}))[0]
    if format == "gptq":
        for bits in (2, 3, 4, 5, 6, 7, 8):
            for group_size in (16, 32, 64, 128, 256, 512, 1024):
                assert validate_quant_linear(
                    linear_class, **(contract | {"bits": bits, "group_size": group_size,
                                               "in_features": max(128, group_size)})
                )[0]
                assert select_quant_linear(
                    bits=bits, group_size=group_size, desc_act=False, sym=False,
                    pack_dtype=torch.int32, dtype=torch.float16, device=DEVICE.MPS,
                    backend=BACKEND.MLX, format=FORMAT.GPTQ_P, quant_method=METHOD.GPTQ,
                ) is MlxQuantLinear
    for change in (
        *(({"bits": 2}, {"bits": 3}, {"bits": 5}, {"bits": 6}, {"bits": 8}) if format == "awq" else ()),
        *(({"bits": 7}, {"group_size": 256}) if format == "awq" else ()),
        {"group_size": 16, "in_features": 112},
        {"desc_act": True},
        {"pack_dtype": torch.int16}, {"dtype": torch.float32},
        {"dtype": torch.bfloat16},
        {"in_features": 120}, {"out_features": 63}, {"device": DEVICE.CPU},
    ):
        assert not validate_quant_linear(linear_class, **(contract | change))[0]


def test_mlx_generate_maps_sampling_options(monkeypatch):
    from gptqmodel.utils import mlx as mlx_utils

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return "generated"

    monkeypatch.setattr(mlx_utils, "generate", fake_generate)
    result = mlx_utils.mlx_generate(
        model=object(), tokenizer=object(), prompt="Paris",
        max_tokens=4, temperature=0.0, top_p=0.9,
        repetition_penalty=1.1, repetition_context_size=32,
    )
    assert result == "generated"
    assert callable(captured["sampler"])
    assert len(captured["logits_processors"]) == 1
    assert "temperature" not in captured and "repetition_penalty" not in captured


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX Metal requires macOS")
def test_auto_selects_mlx_only_for_compatible_models():
    import torch

    from gptqmodel.models._const import DEVICE
    from gptqmodel.models.loader import _auto_select_mlx_backend
    from gptqmodel.quantization.config import FORMAT, METHOD
    from gptqmodel.utils.backend import BACKEND

    class Config:
        def to_dict(self):
            return {"model_type": "qwen2"}

    qcfg = SimpleNamespace(
        bits=4, pack_dtype=torch.int32, group_size=128, sym=False,
        desc_act=False, dynamic=None, rotation=None,
    )

    def select(backend=BACKEND.AUTO, device=DEVICE.MPS, method=METHOD.GPTQ,
               format_code=FORMAT.GPTQ_V2):
        return _auto_select_mlx_backend(
            backend, device, Config(), qcfg, method, format_code, None
        )

    assert select() == BACKEND.MLX
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMM) == BACKEND.MLX
    assert select(backend=BACKEND.GPTQ_TORCH) == BACKEND.GPTQ_TORCH
    assert select(device=DEVICE.CPU) == BACKEND.AUTO
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMV) == BACKEND.AUTO
    qcfg.desc_act = True
    assert select() == BACKEND.AUTO
    qcfg.desc_act = False
    qcfg.group_size = 256
    assert select() == BACKEND.MLX
    qcfg.group_size = 128
    for bits in (2, 3, 4, 5, 6, 7, 8):
        qcfg.bits = bits
        assert select() == BACKEND.MLX
        assert select(format_code=FORMAT.GPTQ_P) == BACKEND.MLX
        assert select(method=METHOD.AWQ, format_code=FORMAT.GEMM) == (BACKEND.MLX if bits == 4 else BACKEND.AUTO)
    qcfg.bits = 4
    qcfg.group_size = -1
    assert select() == BACKEND.MLX
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMM) == BACKEND.MLX
    qcfg.group_size = 16
    assert select() == BACKEND.MLX
    assert select(method=METHOD.AWQ, format_code=FORMAT.GEMM) == BACKEND.MLX
