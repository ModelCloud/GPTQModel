# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from gptqmodel.nn_modules.qlinear.awq_torch import AwqTorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqQuantLinear
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


def pack_awq(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
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


@pytest.mark.skipif(not TORCH_HAS_FUSED_OPS, reason="Torch fused ops require PyTorch>=2.8")
@pytest.mark.parametrize("dtype", [torch.float16], ids=["float16"])
def test_torch_fused_awq_matches_baseline_torch_kernel(dtype):
    torch.manual_seed(0)

    bits = 4
    in_features = 64
    out_features = 128
    group_size = 32
    batch = 4

    groups = in_features // group_size

    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 1.5) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    qweight = pack_awq(int_weight, bits)
    qzeros = pack_awq(zero_points, bits)

    awq_module = AwqTorchQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    awq_module.qweight.copy_(qweight)
    awq_module.qzeros.copy_(qzeros)
    awq_module.scales.copy_(scales)
    awq_module.bias.copy_(bias)
    awq_module.post_init()
    awq_module.eval()

    fused_module = TorchFusedAwqQuantLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    )
    fused_module.register_buffer("qweight", qweight.clone(), persistent=True)
    fused_module.qzeros.copy_(qzeros)
    fused_module.scales.copy_(scales)
    fused_module.bias.copy_(bias)
    fused_module.post_init()
    fused_module.eval()

    x = torch.randn(batch, in_features, dtype=dtype)
    baseline = awq_module(x.to(torch.float16)).to(dtype)
    fused_out = fused_module(x)

    tol_map = {
        torch.float16: 5e-3,
        torch.bfloat16: 1.1,
    }
    tol = tol_map[dtype]
    abs_diff = (fused_out - baseline).abs()
    rel_diff = abs_diff / baseline.abs().clamp_min(1e-6)

    torch.testing.assert_close(fused_out, baseline, rtol=tol, atol=tol)

    header = f"{'dtype':<10} {'rtol':<10} {'atol':<10} {'abs_max':<12} {'rel_max':<12}"
    row = f"{str(dtype):<10} {tol:<10.4g} {tol:<10.4g} {abs_diff.max().item():<12.4e} {rel_diff.max().item():<12.4e}"
    print(f"{header}\n{row}")
