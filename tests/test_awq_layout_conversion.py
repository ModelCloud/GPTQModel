# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from gptqmodel.quantization.awq.modules.linear.gemm import WQLinear_GEMM


def _pack_weights_v1(intweight: torch.Tensor, bits: int) -> torch.Tensor:
    pack_num = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
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
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
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
    scale_zeros = zeros.to(torch.float32) * scales.to(torch.float32)

    linear = torch.nn.Linear(in_features, out_features, bias=True).to(torch.float16)
    with torch.no_grad():
        for idx in range(in_features):
            g = idx // group_size
            weight_col = intweight[:, idx].to(torch.float32) * scales[g] - scale_zeros[g]
            linear.weight[:, idx] = weight_col.to(torch.float16)
        linear.bias.zero_()

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
    module.register_buffer("qweight", qweight_v1.clone())
    module.register_buffer("qzeros", qzeros_v1.clone())
    module.register_buffer("scales", scales_v1.clone())
    module.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))

    module.post_init()

    expected = WQLinear_GEMM.from_linear(
        linear,
        w_bit=bits,
        group_size=group_size,
        scales=scales.t().contiguous(),
        zeros=zeros.t().contiguous(),
    )

    assert module.qweight.dtype == torch.int16
    torch.testing.assert_close(module.qweight.to(torch.int32), expected.qweight.to(torch.int32))
    torch.testing.assert_close(module.scales.to(torch.float32), expected.scales.to(torch.float32))
    torch.testing.assert_close(module.qzeros.to(torch.float32), expected.qzeros.to(torch.float32))
    torch.testing.assert_close(module.bias.to(torch.float16), expected.bias.to(torch.float16))
