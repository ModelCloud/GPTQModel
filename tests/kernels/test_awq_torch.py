# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_matches_manual_dequant(dtype):
    if dtype not in AwqTorchLinear.SUPPORTS_DTYPES:
        pytest.skip(f"dtype {dtype} not supported by AwqTorchLinear")
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
    scales = ((torch.rand(groups, pack_cols, dtype=torch.float32) * 2.0) + 0.25).to(dtype)
    bias = torch.randn(out_features, dtype=dtype)

    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
        register_buffers=True,
    )

    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales.to(module.scales.dtype))
    module.bias.copy_(bias.to(module.bias.dtype))
    module.post_init()
    module.eval()

    batch = 4
    x = torch.randn(batch, in_features, dtype=dtype)

    output_first = module(x)
    output_second = module(x)

    bias_expected = module.bias.to(dtype=dtype)
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(dtype=dtype)
    expected = torch.matmul(x.to(dtype), dequant_weight)
    expected = expected + bias_expected

    assert output_first.dtype == dtype
    assert output_second.dtype == dtype

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
    assert qlinear_cls is AwqTorchLinear
