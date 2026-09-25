# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Shape source: Qwen/Qwen3.8-27B, Apache-2.0, pinned in qwen38_27b_shapes.py.
# MLX runtime: Apple Inc., MIT, https://github.com/ml-explore/mlx
"""Full Qwen3.8-27B projection shape accuracy for packed AWQ GEMV on MLX."""

import gc

import numpy as np
import pytest
import torch

from qwen38_27b_shapes import QWEN38_27B_PROJECTIONS


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def patterned_awq_gemv_weights(input_dims, output_dims):
    """Create varied, deterministic INT4 codes directly in checkpoint words."""
    phase = np.arange(output_dims, dtype=np.uint32)[:, None] % 16
    index = np.arange(input_dims // 8, dtype=np.uint32)[None, :] * 8
    words = np.zeros((output_dims, input_dims // 8), dtype=np.uint32)
    for lane in range(8):
        words |= ((index + phase + lane) % 16) << (lane * 4)
    return words.view(np.int32)


@pytest.mark.parametrize("name,output_dims,input_dims,group_size",
                         [(*shape, 128) for shape in QWEN38_27B_PROJECTIONS]
                         + [("full_attn.k_proj", 1024, 5120, -1)])
def test_qwen38_projection_awq_gemv_accuracy(name, output_dims, input_dims, group_size):
    from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
    from gptqmodel.nn_modules.qlinear.mlx import AwqGemvMlxQuantLinear

    torch.manual_seed(380027)
    source_class = AwqGemvMlxQuantLinear if group_size == -1 else AwqGEMVLinear
    source = source_class(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16,
    )
    words = patterned_awq_gemv_weights(input_dims, output_dims)
    source.qweight.copy_(torch.from_numpy(words))
    source.qzeros.fill_(np.array(0x88888888, dtype=np.uint32).view(np.int32).item())
    source.scales.fill_(0.01)
    assert AwqGemvMlxQuantLinear.source_compatible(source), name
    weight, scales, biases, params = AwqGemvMlxQuantLinear.pack_source(source)
    np.testing.assert_array_equal(weight.view(np.int32), words)
    np.testing.assert_array_equal(scales, np.full_like(scales, np.float16(0.01)))
    np.testing.assert_allclose(biases, -8 * scales.astype(np.float32), rtol=0, atol=1e-6)
    layer = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    layer.load_weights([
        ("weight", mx.array(weight)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    pattern = torch.remainder(
        torch.arange(input_dims)[None, :] + torch.arange(16)[:, None], 16,
    ).float().sub_(8).mul_(float(np.float16(0.01)))
    expected_by_phase = (x.float() @ pattern.T).half().numpy()
    expected = expected_by_phase[:, np.arange(output_dims) % 16]
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002, err_msg=name)
    del layer, source, weight, words
    gc.collect()


@pytest.mark.parametrize("name,output_dims,input_dims", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("format", ["gptq", "awq"])
def test_qwen38_projection_merged_gptq_awq_regression(format, name, output_dims, input_dims):
    """Exercise the already merged GPTQ and AWQ GEMM packed Metal kernels."""
    from gptqmodel.nn_modules.qlinear.mlx import AwqMlxQuantLinear, MlxQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import FORMAT

    torch.manual_seed(380027)
    holder = MlxQuantLinear if format == "gptq" else AwqMlxQuantLinear
    source_class = TorchLinear if format == "gptq" else AwqTorchLinear
    source = source_class(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=input_dims, out_features=output_dims, bias=False,
        register_buffers=True, dtype=torch.float16,
        format=FORMAT.GPTQ_V2 if format == "gptq" else FORMAT.GEMM,
    )
    source.qweight.fill_(0x76543210)
    source.qzeros.fill_(0x11111111)
    source.scales.fill_(0.01)
    if format == "gptq":
        source.qzero_format(2)
    assert holder.source_compatible(source), (format, name)
    weight, scales, biases, params = holder.pack_source(source)
    layer = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    layer.load_weights([
        ("weight", mx.array(weight)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    if format == "gptq":
        # The source word encodes input codes 0..7 in every block.
        code = torch.arange(input_dims).remainder(8).float() - 1
        expected_column = (x.float() * code).sum(dim=-1) * float(np.float16(0.01))
        expected = expected_column[:, None].expand(-1, output_dims).half().numpy()
        assert np.all(weight == np.uint32(0x76543210))
    else:
        # AWQ GEMM reads output nibbles in 0,4,1,5,2,6,3,7 order.
        codes = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32)
        output_codes = np.resize(codes, output_dims)
        expected = ((x.float().sum(dim=-1, keepdim=True).numpy()
                     * (output_codes[None, :].astype(np.float32) - 1)
                     * float(np.float16(0.01))).astype(np.float16))
        np.testing.assert_array_equal(weight[:, 0], output_codes * np.uint32(0x11111111))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002,
                               err_msg=f"{format}: {name}")
    del layer, source, weight
    gc.collect()


def _constant_gptq_words(code, bits, planar):
    """Pack 32 source codes without calling the converter under test."""
    words = np.zeros(bits, dtype=np.uint32)
    planes = {3: ((2, 0), (1, 2)), 5: ((4, 0), (1, 4)),
              6: ((4, 0), (2, 4)), 7: ((4, 0), (2, 4), (1, 6))}
    for index in range(32):
        if planar:
            start = 0
            for width, offset in planes[bits]:
                word = start + index // (32 // width)
                shift = (index % (32 // width)) * width
                words[word] |= np.uint32(((code >> offset) & ((1 << width) - 1)) << shift)
                start += width
        else:
            bit = index * bits
            words[bit // 32] |= np.uint32((code << (bit % 32)) & 0xffffffff)
            if bit % 32 + bits > 32:
                words[bit // 32 + 1] |= np.uint32(code >> (32 - bit % 32))
    return words


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_qwen38_k_projection_merged_gptq_all_bit_rates(bits):
    from gptqmodel.utils.mlx_packing import repack_gptq

    output_dims, input_dims = 1024, 5120  # Qwen3.8-27B full_attn.k_proj
    groups = input_dims // 128
    planar = bits in (5, 6, 7)
    code = (1 << bits) - 1
    source_words = _constant_gptq_words(code, bits, planar)
    zero_words = _constant_gptq_words(code - 1, bits, planar)
    qweight = np.broadcast_to(np.tile(source_words, input_dims // 32)[:, None],
                              (input_dims * bits // 32, output_dims)).copy()
    qzeros = np.broadcast_to(np.tile(zero_words, output_dims // 32)[None, :],
                             (groups, output_dims * bits // 32)).copy()
    scales = np.full((groups, output_dims), np.float16(0.01))
    packed, mlx_scales, biases = repack_gptq(
        qweight, qzeros, scales, input_dims, output_dims, bits, planar,
    )
    target_bits = 8 if bits == 7 else bits
    target_words = _constant_gptq_words(code, target_bits, False)
    np.testing.assert_array_equal(packed[0, :target_bits], target_words)
    layer = nn.QuantizedLinear(input_dims, output_dims, bias=False,
                               group_size=128, bits=target_bits)
    layer.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(mlx_scales)),
        ("biases", mx.array(biases)),
    ])
    torch.manual_seed(380027 + bits)
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    expected = (x.float().sum(dim=-1, keepdim=True) * float(np.float16(0.01)))
    expected = expected.expand(-1, output_dims).half().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("name,output_dims,input_dims", QWEN38_27B_PROJECTIONS)
@pytest.mark.parametrize("bits", [4, 8])
def test_qwen38_projection_qqq_dynamic_input_accuracy(name, output_dims, input_dims, bits):
    from gptqmodel.nn_modules.qlinear.mlx_qqq import MlxQQQLinear

    phase = np.arange(output_dims, dtype=np.uint8)[:, None] % 16
    index = np.arange(input_dims, dtype=np.uint8)[None, :] % 16
    codes = ((index + phase) % 16).astype(np.uint8)
    if bits == 4:
        packed = np.zeros((output_dims, input_dims // 8), dtype=np.uint32)
        lanes = codes.reshape(output_dims, input_dims // 8, 8)
        for lane in range(8):
            packed |= lanes[:, :, lane].astype(np.uint32) << (4 * lane)
    else:
        codes += 120
        packed = np.ascontiguousarray(codes).view(np.uint32).reshape(output_dims, input_dims // 4)
    native = nn.QuantizedLinear(input_dims, output_dims, bias=False,
                                group_size=128, bits=bits)
    native.load_weights([
        ("weight", mx.array(packed)),
        ("scales", mx.full((output_dims, input_dims // 128), 16 if bits == 4 else 1, dtype=mx.float32)),
        ("biases", mx.full((output_dims, input_dims // 128), -128, dtype=mx.float32)),
    ])
    layer = MlxQQQLinear(native, np.full((1, output_dims), 0.001, dtype=np.float32))
    if bits == 8:
        np.testing.assert_array_equal(np.asarray(layer.linear.weight).view(np.uint8).reshape(codes.shape), codes)
    else:
        np.testing.assert_array_equal(np.asarray(layer.linear.weight), packed)
    torch.manual_seed(380027)
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    input_scale = (x.abs().amax(dim=-1, keepdim=True) / 127).float()
    quantized = (x / input_scale).round().clamp(-128, 127).float()
    pattern = (torch.arange(input_dims)[None, :] + torch.arange(16)[:, None]).remainder(16).float() - 8
    expected_by_phase = ((quantized @ pattern.T) * (16 if bits == 4 else 1)
                         * input_scale * 0.001).half().numpy()
    expected = expected_by_phase[:, np.arange(output_dims) % 16]
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002, err_msg=name)


@pytest.mark.parametrize("name,output_dims,input_dims", QWEN38_27B_PROJECTIONS)
def test_qwen38_projection_gguf_q4_0_packed_inference(name, output_dims, input_dims):
    from gptqmodel.utils.mlx_gguf_packing import repack_gguf_affine

    block = np.zeros((1, 1, 18), dtype=np.uint8)
    block[..., :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    block[..., 2:] = 0x98
    source = np.tile(block, (output_dims, input_dims // 32, 1)).reshape(output_dims, -1)
    packed, scales, biases, params = repack_gguf_affine(source, "Q4_0", input_dims)
    np.testing.assert_array_equal(
        packed.reshape(output_dims, -1, 4),
        np.broadcast_to(np.array([0x88888888, 0x88888888, 0x99999999, 0x99999999],
                                 dtype=np.uint32), (output_dims, input_dims // 32, 4)),
    )
    layer = nn.QuantizedLinear(input_dims, output_dims, bias=False, **params)
    layer.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(scales)),
        ("biases", mx.array(biases)),
    ])
    torch.manual_seed(380027)
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    expected_column = x.float().reshape(3, input_dims // 32, 32)[:, :, 16:].sum(dim=(1, 2))
    expected = (expected_column * float(np.float16(0.01)))[:, None].expand(-1, output_dims).half().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002, err_msg=name)


@pytest.mark.parametrize("name,output_dims,input_dims", QWEN38_27B_PROJECTIONS)
def test_qwen38_projection_awq_gemv_fast_llm_layout(name, output_dims, input_dims):
    """GEMV_FAST and LLM_AWQ share this four-output-row packed layout."""
    from gptqmodel.utils.mlx_packing import repack_awq_gemv_fast

    source = np.full((output_dims // 4, input_dims), 0x3210, dtype=np.int16)
    group_count = input_dims // 128
    scales = np.full((group_count, output_dims), np.float16(0.01))
    scaled_zeros = -scales
    packed, mlx_scales, biases = repack_awq_gemv_fast(
        source, scaled_zeros, scales, input_dims, output_dims, 128,
    )
    input_codes = np.arange(input_dims, dtype=np.uint32) // 8 % 4
    np.testing.assert_array_equal(
        packed,
        np.broadcast_to((np.arange(input_dims // 8, dtype=np.uint32) % 4
                         * np.uint32(0x11111111))[None, :], packed.shape),
    )
    layer = nn.QuantizedLinear(input_dims, output_dims, bias=False,
                               group_size=128, bits=4)
    layer.load_weights([
        ("weight", mx.array(packed)), ("scales", mx.array(mlx_scales)),
        ("biases", mx.array(biases)),
    ])
    torch.manual_seed(380027)
    x = (torch.randn(3, input_dims) * 0.05).half()
    actual = layer(mx.array(x.numpy()))
    mx.eval(actual)
    expected_column = x.float() @ torch.from_numpy(input_codes.astype(np.float32) - 1)
    expected = (expected_column[:, None] * float(np.float16(0.01)))
    expected = expected.expand(-1, output_dims).half().numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.002, atol=0.002, err_msg=name)
