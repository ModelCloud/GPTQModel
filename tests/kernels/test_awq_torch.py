# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


@pytest.mark.parametrize("dtype", [torch.float16])
def test_awq_torch_matches_manual_dequant(dtype):
    if dtype not in AwqTorchQuantLinear.SUPPORTS_DTYPES:
        pytest.skip(f"dtype {dtype} not supported by AwqTorchQuantLinear")
    torch.manual_seed(0)

    bits = 4
    in_features = 32
    out_features = 64
    group_size = 16

    assert out_features % (32 // bits) == 0
    assert in_features % group_size == 0

    groups = in_features // group_size
    pack_cols = out_features

    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, pack_cols), dtype=torch.int32)
    scales = (torch.rand(groups, pack_cols, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = AwqTorchQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )

    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales = module.scales.to(dtype=torch.float16)
    module.scales.copy_(scales.to(torch.float16))
    module.bias.copy_(bias)
    module.post_init()
    module.eval()

    batch = 4
    x = torch.randn(batch, in_features, dtype=dtype)

    bias_expected = module.bias

    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
        sym=module.sym,
    ).to(dtype=dtype)

    expected = torch.matmul(x.to(dtype), dequant_weight)
    expected = expected + bias_expected

    output_first = module(x)
    output_second = module(x)

    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-4 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(output_first, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(output_second, expected, atol=atol, rtol=rtol)


def test_awq_torch_backend_selection():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.TORCH_AWQ,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is AwqTorchQuantLinear
