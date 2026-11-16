# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The default KernelConfig must drive AWQ Triton dispatch without an explicit plan."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization.awq.modules.triton.scheduler import default_schedule_mode
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, get_kernel_config


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


def _make_packed_buffers(bits: int, in_features: int, out_features: int, group_size: int):
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.5) + 0.75
    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AWQ Triton forward")
def test_default_kernel_config_forward_uses_configured_schedule_mode():
    pytest.importorskip("triton")
    torch.manual_seed(0)

    previous = get_kernel_config()
    configure_kernel(KernelConfig())
    try:
        assert default_schedule_mode() == "auto"

        bits = 4
        in_features = 512
        out_features = 512
        group_size = 128
        qweight, qzeros, scales = _make_packed_buffers(
            bits, in_features, out_features, group_size
        )

        # No kernel/schedule argument: the module must follow the process-wide
        # default KernelConfig instead of treating its mode as an explicit plan.
        module = AwqGEMMTritonLinear(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        ).cuda()
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.post_init()
        module.eval()

        x = torch.randn(1, 1, in_features, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            actual = module(x)

        assert actual.shape == (1, 1, out_features)
        assert torch.isfinite(actual).all()

        dequant_weight = dequantize_gemm(
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bits=bits,
            group_size=group_size,
        ).to(device=x.device, dtype=x.dtype)
        expected = torch.matmul(x.reshape(-1, in_features), dequant_weight).reshape_as(actual)
        abs_diff = (actual - expected).abs()
        assert abs_diff.max().item() <= 1.0
        assert abs_diff.mean().item() <= 0.02

        # The configured mode belongs to the KernelConfig; the module itself
        # must not materialize it as an explicit schedule.
        assert module.awq_triton_schedule is None
        assert module.awq_triton_schedule_mode is None
    finally:
        configure_kernel(previous)
