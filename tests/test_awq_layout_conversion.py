# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import pytest

from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from gptqmodel.quantization.awq.utils.mit_repacker import (
    multiply_scale_qzero_negative as mit_multiply_scale_qzero_negative,
    packing_v2_from_unpacked as mit_packing_v2_from_unpacked,
    qweight_unpack as mit_qweight_unpack,
)
from gptqmodel.quantization.awq.utils.module import try_import
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.awq.modules.linear.gemv_fast import calculate_zeros_width
from safetensors.torch import load_file


def _pack_weights_v1(intweight: torch.Tensor, bits: int) -> torch.Tensor:
    pack_num = 32 // bits
    order_map = list(range(pack_num))
    rows, cols = intweight.shape
    packed_cols = (cols + pack_num - 1) // pack_num
    packed = torch.zeros((rows, packed_cols), dtype=torch.int32)
    mask = (1 << bits) - 1
    for col in range(packed_cols):
        for idx, order in enumerate(order_map):
            src = col * pack_num + order
            if src >= cols:
                continue
            packed[:, col] |= ((intweight[:, src].to(torch.int32) & mask) << (idx * bits))
    return packed


def _pack_zeros_v1(zeros: torch.Tensor, bits: int) -> torch.Tensor:
    pack_num = 32 // bits
    order_map = list(range(pack_num))
    rows, cols = zeros.shape
    packed_cols = (cols + pack_num - 1) // pack_num
    packed = torch.zeros((rows, packed_cols), dtype=torch.int32)
    mask = (1 << bits) - 1
    for col in range(packed_cols):
        for idx, order in enumerate(order_map):
            src = col * pack_num + order
            if src >= cols:
                continue
            packed[:, col] |= ((zeros[:, src].to(torch.int32) & mask) << (idx * bits))
    return packed


def test_awq_gemm_legacy_conversion_matches_v2():
    torch.manual_seed(0)
    bits = 4
    group_size = 128
    in_features = 256
    out_features = 128

    groups = in_features // group_size
    assert groups * group_size == in_features

    intweight = torch.randint(0, 2 ** bits, size=(out_features, in_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features, dtype=torch.float16) * 2.0 + 0.5
    zeros = torch.randint(0, 2 ** bits, size=(groups, out_features), dtype=torch.int32)
    intweight_t = intweight.t().contiguous()
    qweight_v1 = _pack_weights_v1(intweight_t, bits)
    qzeros_v1 = _pack_zeros_v1(zeros, bits)
    scales_v1 = scales.clone()

    module = AwqGEMMQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=False,
    )
    module.load_legacy_tensors(
        qweight_v1.clone(),
        qzeros_v1.clone(),
        scales_v1.clone(),
        torch.zeros(out_features, dtype=torch.float16),
    )

    unpacked_expected = mit_qweight_unpack(qweight_v1.clone())
    if unpacked_expected.shape[0] == in_features and unpacked_expected.shape[1] == out_features:
        unpacked_expected = unpacked_expected.transpose(0, 1).contiguous()
    expected_qweight = mit_packing_v2_from_unpacked(unpacked_expected, interleave=4, kstride=64)
    pack_num = 32 // bits
    zeros_width = calculate_zeros_width(in_features, group_size, pack_num=pack_num)
    expected_scales = torch.zeros((zeros_width * pack_num, out_features), dtype=torch.float16)
    expected_scales[: scales.shape[0], :] = scales

    def reference_scaled_zeros(scales_ref: torch.Tensor, qzeros_ref: torch.Tensor, shift: int = 0) -> torch.Tensor:
        pack_size = 8
        rows, cols = scales_ref.shape
        qzeros_ref = qzeros_ref.to(torch.int32)
        col_indices = torch.arange(cols, device=scales_ref.device, dtype=torch.int32)
        zero_idx = col_indices // pack_size
        zero_offset = col_indices % pack_size
        zeros = (qzeros_ref[:, zero_idx] >> (zero_offset * 4)) & 0xF
        zeros = zeros.to(scales_ref.dtype)
        scaled = scales_ref * zeros
        if shift:
            scaled = scaled + shift * scales_ref
        return -scaled

    expected_qzeros = torch.zeros_like(expected_scales)
    expected_qzeros[: scales.shape[0], :] = reference_scaled_zeros(scales, qzeros_v1)
    expected_bias = torch.zeros(out_features, dtype=torch.float16)

    assert module.qweight.dtype == torch.int16
    torch.testing.assert_close(module.qweight.to(torch.int32), expected_qweight.to(torch.int32))
    torch.testing.assert_close(module.scales.to(torch.float32), expected_scales.to(torch.float32))
    torch.testing.assert_close(module.qzeros.to(torch.float32), expected_qzeros.to(torch.float32))
    torch.testing.assert_close(module.bias.to(torch.float16), expected_bias.to(torch.float16))

    awq_ext, _ = try_import("gptqmodel_awq_kernels")
    if awq_ext is None or not torch.cuda.is_available():
        pytest.skip("AWQ CUDA kernels unavailable for forward validation")

    module = module.to("cuda")
    inputs = torch.randn(2, 8, in_features, dtype=torch.float16, device="cuda")

    def dense_weight_from_v1(intweight_ref: torch.Tensor, zeros_ref: torch.Tensor, scales_ref: torch.Tensor) -> torch.Tensor:
        intweight_ref = intweight_ref.to(torch.float32)
        zeros_ref = zeros_ref.to(torch.float32)
        scales_ref = scales_ref.to(torch.float32)
        weight = torch.empty(in_features, out_features, dtype=torch.float32)
        for group_idx in range(groups):
            start = group_idx * group_size
            end = start + group_size
            block = intweight_ref[:, start:end].T  # (group, out_features)
            zero = zeros_ref[group_idx].unsqueeze(0)
            scale = scales_ref[group_idx].unsqueeze(0)
            weight[start:end] = (block - zero) * scale
        return weight

    weight_ref = dense_weight_from_v1(intweight, zeros, scales).to(device="cuda", dtype=torch.float16)
    with torch.no_grad():
        expected_out = torch.matmul(inputs.reshape(-1, in_features), weight_ref)
        expected_out = expected_out.view(inputs.shape[0], inputs.shape[1], out_features)
        actual_out = module(inputs)
    torch.testing.assert_close(actual_out, expected_out, atol=5e-1, rtol=5e-3)


@pytest.mark.cuda
def test_awq_gemm_conversion_real_checkpoint():
    ckpt_dir = Path("/monster/data/model/deepseek-r1-distill-qwen-7b-awq")
    safetensor_path = ckpt_dir / "model-00001-of-00002.safetensors"
    if not safetensor_path.exists():
        pytest.skip("DeepSeek AWQ checkpoint unavailable at expected path.")

    tensors = load_file(str(safetensor_path))
    tensor_prefix = "model.layers.0.self_attn.q_proj"
    qweight = tensors[f"{tensor_prefix}.qweight"]
    qzeros = tensors[f"{tensor_prefix}.qzeros"]
    scales = tensors[f"{tensor_prefix}.scales"]

    bits = 4
    group_size = 128
    groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = groups * group_size

    module = AwqGEMMQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    module.load_legacy_tensors(qweight, qzeros, scales, bias=None)

    pack_num = 32 // bits
    zeros_width = calculate_zeros_width(in_features, group_size, pack_num=pack_num)

    unpacked = mit_qweight_unpack(qweight)
    if unpacked.shape == (in_features, out_features):
        unpacked = unpacked.transpose(0, 1).contiguous()
    expected_qweight = mit_packing_v2_from_unpacked(unpacked, interleave=4, kstride=64)

    scales_groups = scales if scales.shape == (groups, out_features) else scales.transpose(0, 1).contiguous()
    expected_zero_cols = out_features // pack_num
    qzeros_groups = qzeros
    if qzeros_groups.shape == (out_features, expected_zero_cols):
        qzeros_groups = qzeros_groups.transpose(0, 1).contiguous()
    elif qzeros_groups.shape == (expected_zero_cols, out_features):
        qzeros_groups = qzeros_groups.transpose(0, 1).contiguous()

    scaled_zeros_groups = mit_multiply_scale_qzero_negative(scales_groups, qzeros_groups, zp_shift=0)

    padded_rows = zeros_width * pack_num
    expected_scales = torch.zeros((padded_rows, out_features), dtype=scales_groups.dtype)
    expected_zeros = torch.zeros_like(expected_scales)
    expected_scales[: groups, :] = scales_groups
    expected_zeros[: groups, :] = scaled_zeros_groups

    torch.testing.assert_close(module.qweight.to(torch.int32), expected_qweight.to(torch.int32))
    torch.testing.assert_close(module.scales.to(torch.float32), expected_scales.to(torch.float32))
    torch.testing.assert_close(module.qzeros.to(torch.float32), expected_zeros.to(torch.float32))
