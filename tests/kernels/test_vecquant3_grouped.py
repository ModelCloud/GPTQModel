# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import pytest  # noqa: E402
import torch  # noqa: E402

from gptqmodel.adapter.quant import dequantize_tensor_groupwise_int8, quantize_tensor_groupwise_int8  # noqa: E402
from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul as dequant_quant_matmul  # noqa: E402
from gptqmodel.utils import grasshopper, vecquant3  # noqa: E402


def _pack_int3_rows(values: torch.Tensor) -> torch.Tensor:
    in_features, out_features = values.shape
    values = values.to(torch.int64).reshape(in_features // 32, 32, out_features)
    word0 = torch.zeros((in_features // 32, out_features), dtype=torch.int64)
    word1 = torch.zeros_like(word0)
    word2 = torch.zeros_like(word0)
    for i in range(10):
        word0 |= (values[:, i] & 0x7) << (3 * i)
    word0 |= (values[:, 10] & 0x3) << 30
    word1 |= (values[:, 10] >> 2) & 0x1
    for i in range(11, 21):
        word1 |= (values[:, i] & 0x7) << (1 + 3 * (i - 11))
    word1 |= (values[:, 21] & 0x1) << 31
    word2 |= (values[:, 21] >> 1) & 0x3
    for i in range(22, 32):
        word2 |= (values[:, i] & 0x7) << (2 + 3 * (i - 22))
    return torch.stack((word0, word1, word2), dim=1).reshape((in_features // 32) * 3, out_features).to(torch.int32)


def _pack_int3_cols(values: torch.Tensor) -> torch.Tensor:
    groups, out_features = values.shape
    values = values.to(torch.int64).reshape(groups, out_features // 32, 32)
    word0 = torch.zeros((groups, out_features // 32), dtype=torch.int64)
    word1 = torch.zeros_like(word0)
    word2 = torch.zeros_like(word0)
    for i in range(10):
        word0 |= (values[:, :, i] & 0x7) << (3 * i)
    word0 |= (values[:, :, 10] & 0x3) << 30
    word1 |= (values[:, :, 10] >> 2) & 0x1
    for i in range(11, 21):
        word1 |= (values[:, :, i] & 0x7) << (1 + 3 * (i - 11))
    word1 |= (values[:, :, 21] & 0x1) << 31
    word2 |= (values[:, :, 21] >> 1) & 0x3
    for i in range(22, 32):
        word2 |= (values[:, :, i] & 0x7) << (2 + 3 * (i - 22))
    return torch.stack((word0, word1, word2), dim=2).reshape(groups, (out_features // 32) * 3).to(torch.int32)


def _pack_rows(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 3:
        return _pack_int3_rows(values)

    in_features, out_features = values.shape
    pack_factor = 32 // bits
    values = values.to(torch.int64).reshape(in_features // 32, bits, pack_factor, out_features)
    packed = torch.zeros((in_features // 32, bits, out_features), dtype=torch.int64)
    for word in range(bits):
        for offset in range(pack_factor):
            packed[:, word] |= (values[:, word, offset] & ((1 << bits) - 1)) << (bits * offset)
    return packed.reshape((in_features // 32) * bits, out_features).to(torch.int32)


def _pack_cols(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 3:
        return _pack_int3_cols(values)

    groups, out_features = values.shape
    pack_factor = 32 // bits
    values = values.to(torch.int64).reshape(groups, out_features // 32, bits, pack_factor)
    packed = torch.zeros((groups, out_features // 32, bits), dtype=torch.int64)
    for word in range(bits):
        for offset in range(pack_factor):
            packed[:, :, word] |= (values[:, :, word, offset] & ((1 << bits) - 1)) << (bits * offset)
    return packed.reshape(groups, (out_features // 32) * bits).to(torch.int32)


def _base_atol(bits: int, dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 5.0e-1 if bits == 8 else 8e-2
    return 5e-2 if bits == 8 else 3e-2


def _input_accum_atol(bits: int, dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 2.0 if bits == 8 else 2e-1
    return 3e-1 if bits == 8 else 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VecQuant3 grouped kernel test")
@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_features", [96, 256])
def test_vecquant3_grouped_gemv_matches_dequant_reference(bits, group_size, dtype, out_features):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    torch.manual_seed(19 + group_size)
    device = torch.device("cuda:0")
    maxq = (1 << bits) - 1
    pack_bits = 32
    in_features = 256
    rank = 64
    groups = in_features // group_size

    x = torch.randn(in_features, device=device, dtype=torch.float16)
    qweight_values = torch.randint(0, maxq + 1, (in_features, out_features), dtype=torch.int64)
    qzero_values = torch.randint(0, maxq + 1, (groups, out_features), dtype=torch.int64)
    qweight = _pack_rows(qweight_values, bits).to(device)
    qzeros = _pack_cols(qzero_values, bits).to(device)
    scales = (torch.rand(groups, out_features, device=device, dtype=dtype) * 0.02 + 0.001).contiguous()
    g_idx = (torch.arange(in_features, device=device, dtype=torch.int32) // group_size).contiguous()
    x = x.to(dtype=dtype)

    expected = dequant_quant_matmul(x.reshape(1, -1), qweight, scales, qzeros, g_idx, bits, pack_bits, maxq).reshape(-1).float()
    actual = vecquant3.gemv(x, qweight, scales, qzeros, group_size, accumulation_dtype=torch.float32, bits=bits)
    float_atol = _base_atol(bits, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=float_atol)

    actual_input_accum = vecquant3.gemv(x, qweight, scales, qzeros, group_size, accumulation_dtype=dtype, bits=bits)
    input_atol = _input_accum_atol(bits, dtype)
    torch.testing.assert_close(actual_input_accum, expected, rtol=0, atol=input_atol)

    lora_a = (torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01).contiguous()
    lora_b = (torch.randn(rank, out_features, device=device, dtype=dtype) * 0.01).contiguous()
    down = (x.reshape(1, -1) @ lora_a).reshape(-1).contiguous()
    lora_term = (down.reshape(-1, 1) * lora_b).float().sum(dim=0)

    expected_lora = actual + lora_term
    actual_lora = vecquant3.gemv_lora(
        x, qweight, scales, qzeros, down, lora_b, group_size, accumulation_dtype=torch.float32, bits=bits
    )
    lora_atol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual_lora, expected_lora, rtol=0, atol=lora_atol)

    expected_lora_input_accum = actual_input_accum + lora_term
    actual_lora_input_accum = vecquant3.gemv_lora(
        x, qweight, scales, qzeros, down, lora_b, group_size, accumulation_dtype="input", bits=bits
    )
    torch.testing.assert_close(actual_lora_input_accum, expected_lora_input_accum, rtol=0, atol=lora_atol)

    for lora_group_size in (32, 64, 96, 128):
        up_qweight, up_scales, up_shape = quantize_tensor_groupwise_int8(
            lora_b,
            group_size=lora_group_size,
            scale_dtype=dtype,
        )
        up_dequant = dequantize_tensor_groupwise_int8(
            qweight=up_qweight,
            scales=up_scales,
            shape=up_shape,
            group_size=lora_group_size,
            device=device,
            dtype=dtype,
        ).contiguous()
        up_qweight = up_qweight.to(device=device, non_blocking=True).contiguous()
        up_scales = up_scales.to(device=device, non_blocking=True).contiguous()
        lora_int8_term = (down.reshape(-1, 1) * up_dequant).float().sum(dim=0)

        expected_lora_int8 = actual + lora_int8_term
        actual_lora_int8 = vecquant3.gemv_lora_int8(
            x,
            qweight,
            scales,
            qzeros,
            down,
            up_qweight,
            up_scales,
            up_shape,
            group_size,
            lora_group_size,
            accumulation_dtype=torch.float32,
            bits=bits,
        )
        int8_lora_atol = 8e-3 if dtype == torch.float16 else 3e-2
        torch.testing.assert_close(actual_lora_int8, expected_lora_int8, rtol=0, atol=int8_lora_atol)

        expected_lora_int8_input_accum = actual_input_accum + lora_int8_term
        actual_lora_int8_input_accum = vecquant3.gemv_lora_int8(
            x,
            qweight,
            scales,
            qzeros,
            down,
            up_qweight,
            up_scales,
            up_shape,
            group_size,
            lora_group_size,
            accumulation_dtype="input",
            bits=bits,
        )
        torch.testing.assert_close(
            actual_lora_int8_input_accum,
            expected_lora_int8_input_accum,
            rtol=0,
            atol=int8_lora_atol,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VecQuant3 grouped kernel test")
@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_features", [96, 256])
def test_vecquant3_grouped_gemm_matches_dequant_reference(bits, group_size, dtype, out_features):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    torch.manual_seed(37 + group_size)
    device = torch.device("cuda:0")
    maxq = (1 << bits) - 1
    pack_bits = 32
    batch_size = 3
    in_features = 256
    rank = 64
    groups = in_features // group_size

    x = torch.randn(batch_size, in_features, device=device, dtype=torch.float16).to(dtype=dtype).contiguous()
    qweight_values = torch.randint(0, maxq + 1, (in_features, out_features), dtype=torch.int64)
    qzero_values = torch.randint(0, maxq + 1, (groups, out_features), dtype=torch.int64)
    qweight = _pack_rows(qweight_values, bits).to(device)
    qzeros = _pack_cols(qzero_values, bits).to(device)
    scales = (torch.rand(groups, out_features, device=device, dtype=dtype) * 0.02 + 0.001).contiguous()
    g_idx = (torch.arange(in_features, device=device, dtype=torch.int32) // group_size).contiguous()

    expected = dequant_quant_matmul(x, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq).float()
    actual = vecquant3.gemm(x, qweight, scales, qzeros, group_size, accumulation_dtype=torch.float32, bits=bits)
    float_atol = _base_atol(bits, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=float_atol)

    actual_input_accum = vecquant3.gemm(x, qweight, scales, qzeros, group_size, accumulation_dtype=dtype, bits=bits)
    input_atol = _input_accum_atol(bits, dtype)
    torch.testing.assert_close(actual_input_accum, expected, rtol=0, atol=input_atol)

    lora_a = (torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01).contiguous()
    lora_b = (torch.randn(rank, out_features, device=device, dtype=dtype) * 0.01).contiguous()
    down = (x @ lora_a).contiguous()
    lora_term = (down @ lora_b).float()

    expected_lora = actual + lora_term
    actual_lora = vecquant3.gemm_lora(
        x, qweight, scales, qzeros, down, lora_b, group_size, accumulation_dtype=torch.float32, bits=bits
    )
    lora_atol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual_lora, expected_lora, rtol=0, atol=lora_atol)

    expected_lora_input_accum = actual_input_accum + lora_term
    actual_lora_input_accum = vecquant3.gemm_lora(
        x, qweight, scales, qzeros, down, lora_b, group_size, accumulation_dtype="input", bits=bits
    )
    torch.testing.assert_close(actual_lora_input_accum, expected_lora_input_accum, rtol=0, atol=lora_atol)

    for lora_group_size in (32, 64, 96, 128):
        up_qweight, up_scales, up_shape = quantize_tensor_groupwise_int8(
            lora_b,
            group_size=lora_group_size,
            scale_dtype=dtype,
        )
        up_dequant = dequantize_tensor_groupwise_int8(
            qweight=up_qweight,
            scales=up_scales,
            shape=up_shape,
            group_size=lora_group_size,
            device=device,
            dtype=dtype,
        ).contiguous()
        up_qweight = up_qweight.to(device=device, non_blocking=True).contiguous()
        up_scales = up_scales.to(device=device, non_blocking=True).contiguous()
        lora_int8_term = (down @ up_dequant).float()

        expected_lora_int8 = actual + lora_int8_term
        actual_lora_int8 = vecquant3.gemm_lora_int8(
            x,
            qweight,
            scales,
            qzeros,
            down,
            up_qweight,
            up_scales,
            up_shape,
            group_size,
            lora_group_size,
            accumulation_dtype=torch.float32,
            bits=bits,
        )
        int8_lora_atol = 8e-3 if dtype == torch.float16 else 3e-2
        torch.testing.assert_close(actual_lora_int8, expected_lora_int8, rtol=0, atol=int8_lora_atol)

        expected_lora_int8_input_accum = actual_input_accum + lora_int8_term
        actual_lora_int8_input_accum = vecquant3.gemm_lora_int8(
            x,
            qweight,
            scales,
            qzeros,
            down,
            up_qweight,
            up_scales,
            up_shape,
            group_size,
            lora_group_size,
            accumulation_dtype="input",
            bits=bits,
        )
        torch.testing.assert_close(
            actual_lora_int8_input_accum,
            expected_lora_int8_input_accum,
            rtol=0,
            atol=int8_lora_atol,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for VecQuant3 grouped kernel test")
@pytest.mark.parametrize("bits", [3, 4, 8])
def test_vecquant3_grouped_gemm_tiled_matches_gemv_loop(bits):
    torch.manual_seed(53)
    device = torch.device("cuda:0")
    dtype = torch.float16
    group_size = 128
    lora_group_size = 128
    batch_size = 4
    in_features = 256
    out_features = 2048
    rank = 32
    groups = in_features // group_size

    x = torch.randn(batch_size, in_features, device=device, dtype=dtype).contiguous()
    maxq = (1 << bits) - 1
    qweight_values = torch.randint(0, maxq + 1, (in_features, out_features), dtype=torch.int64)
    qzero_values = torch.randint(0, maxq + 1, (groups, out_features), dtype=torch.int64)
    qweight = _pack_rows(qweight_values, bits).to(device)
    qzeros = _pack_cols(qzero_values, bits).to(device)
    scales = (torch.rand(groups, out_features, device=device, dtype=dtype) * 0.02 + 0.001).contiguous()

    expected = torch.stack(
        [
            vecquant3.gemv(sample, qweight, scales, qzeros, group_size, accumulation_dtype=torch.float32, bits=bits)
            for sample in x
        ]
    )
    actual = vecquant3.gemm(x, qweight, scales, qzeros, group_size, accumulation_dtype=torch.float32, bits=bits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-3)

    lora_a = (torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01).contiguous()
    lora_b = (torch.randn(rank, out_features, device=device, dtype=dtype) * 0.01).contiguous()
    down = (x @ lora_a).contiguous()
    expected_lora = torch.stack(
        [
            vecquant3.gemv_lora(
                x[row],
                qweight,
                scales,
                qzeros,
                down[row],
                lora_b,
                group_size,
                accumulation_dtype=torch.float32,
                bits=bits,
            )
            for row in range(batch_size)
        ]
    )
    actual_lora = vecquant3.gemm_lora(
        x, qweight, scales, qzeros, down, lora_b, group_size, accumulation_dtype=torch.float32, bits=bits
    )
    torch.testing.assert_close(actual_lora, expected_lora, rtol=0, atol=4e-3)

    up_qweight, up_scales, up_shape = quantize_tensor_groupwise_int8(
        lora_b,
        group_size=lora_group_size,
        scale_dtype=dtype,
    )
    up_qweight = up_qweight.to(device=device, non_blocking=True).contiguous()
    up_scales = up_scales.to(device=device, non_blocking=True).contiguous()
    expected_lora_int8 = torch.stack(
        [
            vecquant3.gemv_lora_int8(
                x[row],
                qweight,
                scales,
                qzeros,
                down[row],
                up_qweight,
                up_scales,
                up_shape,
                group_size,
                lora_group_size,
                accumulation_dtype=torch.float32,
                bits=bits,
            )
            for row in range(batch_size)
        ]
    )
    actual_lora_int8 = vecquant3.gemm_lora_int8(
        x,
        qweight,
        scales,
        qzeros,
        down,
        up_qweight,
        up_scales,
        up_shape,
        group_size,
        lora_group_size,
        accumulation_dtype=torch.float32,
        bits=bits,
    )
    torch.testing.assert_close(actual_lora_int8, expected_lora_int8, rtol=0, atol=4e-3)


def test_vecquant3_accumulation_dtype_validation():
    assert vecquant3._normalize_accumulation_dtype(torch.float32, torch.float16) == 0
    assert vecquant3._normalize_accumulation_dtype("input", torch.float16) == 1
    assert vecquant3._normalize_accumulation_dtype("bf16", torch.bfloat16) == 1
    assert vecquant3._normalize_bits(3) == 3
    assert vecquant3._normalize_bits(4) == 4
    assert vecquant3._normalize_bits(8) == 8
    assert grasshopper.SUPPORTED_BITS == (3, 4, 8)
    with pytest.raises(ValueError, match="accumulation_dtype"):
        vecquant3._normalize_accumulation_dtype("bf16", torch.float16)
    with pytest.raises(ValueError, match="3, 4, or 8"):
        vecquant3._normalize_bits(2)
