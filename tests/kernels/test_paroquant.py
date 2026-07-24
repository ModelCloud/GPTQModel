# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant kernel tests adapted from the ParoQuant paper and public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Kernel-focused tests for ParoQuant runtime behavior and backend parity."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import paroquant_triton as paroquant_triton_qlinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.paroquant.optimization import build_random_rotation_buffers
from gptqmodel.quantization.paroquant.modules.triton import gemm as paroquant_triton_gemm
from gptqmodel.quantization.paroquant.modules.triton.gemm import (
    _paroquant_rotation_gemm_splitk_triton,
    _paroquant_rotation_gemm_triton,
    _paroquant_splitk_launch_config,
    _paroquant_splitk_output_config,
    paroquant_gemm_triton_decode,
    paroquant_rotation_gemm_triton_decode,
    paroquant_rotation_gemm_triton_prefill,
)
from gptqmodel.utils.awq import awq_runtime_available
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.paroquant import (
    apply_paroquant_rotation_reference,
    build_identity_rotation_buffers,
    build_paroquant_rotation_lookup,
)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integer weights into the AWQ bit layout used by the kernels."""
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
    """Build synthetic packed AWQ tensors for ParoQuant runtime tests."""
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


def _upstream_transformers_contract_reference(
    x: torch.Tensor,
    *,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    theta: torch.Tensor,
    channel_scales: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
) -> torch.Tensor:
    """Reference for upstream RotateQuantizedLinear.forward().

    Upstream ParoQuant applies per-projection rotation to the input and then
    feeds the rotated activations into the AWQ GEMM path. We reproduce that
    contract with dense dequantization here to assess kernel accuracy without
    importing or copying upstream implementation code.
    """

    rotated = apply_paroquant_rotation_reference(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=group_size,
    )
    dense_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=bits,
        group_size=group_size,
    ).to(device=x.device, dtype=x.dtype)
    out = torch.matmul(rotated.reshape(-1, x.shape[-1]), dense_weight).reshape(*x.shape[:-1], out_features)
    if bias is not None:
        out = out + bias.to(device=x.device, dtype=x.dtype)
    return out


def test_paroquant_identity_forward_matches_awq_torch():
    """Guard that identity ParoQuant is behaviorally equivalent to plain AWQ."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    awq_module = AwqTorchLinear(
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

    paro_module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    )
    paro_module.qweight.copy_(qweight)
    paro_module.qzeros.copy_(qzeros)
    paro_module.scales.copy_(scales)
    paro_module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        dtype=torch.float16,
    )
    paro_module.pairs.copy_(pairs)
    paro_module.theta.copy_(theta)
    paro_module.channel_scales.copy_(channel_scales)
    paro_module.post_init()
    paro_module.eval()

    x = torch.randn(4, in_features, dtype=torch.float16)
    torch.testing.assert_close(paro_module(x), awq_module(x), atol=5e-3, rtol=5e-3)


def test_paroquant_forward_matches_explicit_rotated_reference():
    """Guard the dense reference contract for non-identity ParoQuant rotations."""
    bits = 4
    in_features = 128
    out_features = 64
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=1,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        dtype=torch.float16,
    )
    theta.fill_(0.2)
    channel_scales.mul_(0.75)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(3, in_features, dtype=torch.float16)
    rotated = apply_paroquant_rotation_reference(
        x,
        module.pairs,
        module.theta,
        scales=module.channel_scales,
        group_size=group_size,
    )
    dequant_weight = dequantize_gemm(
        qweight=module.qweight,
        qzeros=module.qzeros,
        scales=module.scales,
        bits=bits,
        group_size=group_size,
    ).to(dtype=x.dtype)
    expected = torch.matmul(rotated, dequant_weight) + module.bias

    torch.testing.assert_close(module(x), expected, atol=5e-3, rtol=5e-3)


def test_paroquant_rotation_lookup_encodes_pair_updates():
    """Guard the gather lookup consumed by the rotation+GEMM megakernel."""
    pairs, theta, _ = build_identity_rotation_buffers(
        in_features=8,
        group_size=4,
        krot=1,
        dtype=torch.float16,
    )
    theta.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=theta.dtype))

    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=4)

    torch.testing.assert_close(
        partner,
        torch.tensor([[1, 0, 3, 2, 5, 4, 7, 6]], dtype=torch.int32),
    )
    angles = theta.float().repeat_interleave(2, dim=1)
    signs = torch.tensor([[1.0, -1.0] * 4])
    torch.testing.assert_close(cos, angles.cos())
    torch.testing.assert_close(sin, angles.sin() * signs)


def test_paroquant_rotation_lookup_rejects_duplicate_features():
    """Reject lookup metadata that would leave output features unwritten."""
    pairs, theta, _ = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=1,
        dtype=torch.float16,
    )
    pairs[0, 1] = pairs[0, 0]

    with pytest.raises(ValueError, match="pair each group feature exactly once"):
        build_paroquant_rotation_lookup(pairs, theta, group_size=128)


def test_paroquant_prefill_routes_measured_row_tiles(monkeypatch):
    """Use narrow and wave-aware wide row tiles only for their measured dtype/shape regimes."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_configs = []

    def capture_config(*_args, **kwargs):
        captured_configs.append(
            (
                kwargs["block_size_m"],
                kwargs["loop_unroll_factor"],
                kwargs["explicit_fma"],
                kwargs["prefetch_first_partner"],
            )
        )
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    metadata = torch.empty((8, 0))
    cases = (
        (torch.empty((128, 2048), dtype=torch.float16), 64),
        (torch.empty((128, 2048), dtype=torch.bfloat16), 64),
        (torch.empty((128, 2048), dtype=torch.float16), 65),
        (torch.empty((128, 2048), dtype=torch.float16), 256),
        (torch.empty((128, 2048), dtype=torch.bfloat16), 256),
        (torch.empty((64, 2048), dtype=torch.float16), 256),
        (torch.empty((128, 512), dtype=torch.float16), 64),
        (torch.empty((128, 1024), dtype=torch.bfloat16), 64),
        (torch.empty((128, 512), dtype=torch.bfloat16), 64),
        (torch.empty((112, 2048), dtype=torch.float16), 256),
        (torch.empty((113, 2048), dtype=torch.float16), 256),
        (torch.empty((224, 2048), dtype=torch.float16), 256),
        (torch.empty((225, 2048), dtype=torch.float16), 256),
        (torch.empty((368, 2048), dtype=torch.float16), 256),
        (torch.empty((369, 2048), dtype=torch.float16), 256),
        (torch.empty((480, 2048), dtype=torch.float16), 256),
        (torch.empty((481, 2048), dtype=torch.float16), 256),
        (torch.empty((608, 2048), dtype=torch.float16), 256),
        (torch.empty((609, 2048), dtype=torch.float16), 256),
        (torch.empty((736, 2048), dtype=torch.float16), 256),
        (torch.empty((737, 2048), dtype=torch.float16), 256),
        (torch.empty((864, 2048), dtype=torch.float16), 256),
        (torch.empty((865, 2048), dtype=torch.float16), 256),
        (torch.empty((992, 2048), dtype=torch.float16), 256),
        (torch.empty((993, 2048), dtype=torch.float16), 256),
        (torch.empty((240, 2048), dtype=torch.float16), 128),
        (torch.empty((241, 2048), dtype=torch.float16), 128),
        (torch.empty((480, 2048), dtype=torch.float16), 128),
        (torch.empty((481, 2048), dtype=torch.float16), 128),
        (torch.empty((160, 2048), dtype=torch.float16), 192),
        (torch.empty((161, 2048), dtype=torch.float16), 192),
        (torch.empty((320, 2048), dtype=torch.float16), 192),
        (torch.empty((321, 2048), dtype=torch.float16), 192),
        (torch.empty((320, 2048), dtype=torch.float16), 96),
        (torch.empty((321, 2048), dtype=torch.float16), 96),
        (torch.empty((640, 2048), dtype=torch.float16), 96),
        (torch.empty((641, 2048), dtype=torch.float16), 96),
        (torch.empty((192, 2048), dtype=torch.float16), 160),
        (torch.empty((193, 2048), dtype=torch.float16), 160),
        (torch.empty((384, 2048), dtype=torch.float16), 160),
        (torch.empty((385, 2048), dtype=torch.float16), 160),
        (torch.empty((128, 2048), dtype=torch.float16), 224),
        (torch.empty((129, 2048), dtype=torch.float16), 224),
        (torch.empty((256, 2048), dtype=torch.float16), 224),
        (torch.empty((257, 2048), dtype=torch.float16), 224),
        (torch.empty((496, 2048), dtype=torch.float16), 64),
        (torch.empty((497, 2048), dtype=torch.float16), 64),
        (torch.empty((992, 2048), dtype=torch.float16), 64),
        (torch.empty((993, 2048), dtype=torch.float16), 64),
        (torch.empty((496, 2048), dtype=torch.bfloat16), 64),
        (torch.empty((497, 2048), dtype=torch.bfloat16), 64),
        (torch.empty((992, 2048), dtype=torch.bfloat16), 64),
        (torch.empty((993, 2048), dtype=torch.bfloat16), 64),
        (torch.empty((992, 2048), dtype=torch.float16), 32),
        (torch.empty((993, 2048), dtype=torch.float16), 32),
        (torch.empty((1984, 2048), dtype=torch.float16), 32),
        (torch.empty((1985, 2048), dtype=torch.float16), 32),
        (torch.empty((992, 2048), dtype=torch.bfloat16), 32),
        (torch.empty((993, 2048), dtype=torch.bfloat16), 32),
        (torch.empty((1984, 2048), dtype=torch.bfloat16), 32),
        (torch.empty((1985, 2048), dtype=torch.bfloat16), 32),
        (torch.empty((1984, 2048), dtype=torch.float16), 16),
        (torch.empty((1985, 2048), dtype=torch.float16), 16),
        (torch.empty((3968, 2048), dtype=torch.float16), 16),
        (torch.empty((3969, 2048), dtype=torch.float16), 16),
        (torch.empty((1984, 2048), dtype=torch.bfloat16), 16),
        (torch.empty((1985, 2048), dtype=torch.bfloat16), 16),
        (torch.empty((3968, 2048), dtype=torch.bfloat16), 16),
        (torch.empty((3969, 2048), dtype=torch.bfloat16), 16),
        (torch.empty((656, 2048), dtype=torch.float16), 48),
        (torch.empty((657, 2048), dtype=torch.float16), 48),
        (torch.empty((1312, 2048), dtype=torch.float16), 48),
        (torch.empty((1313, 2048), dtype=torch.float16), 48),
        (torch.empty((656, 2048), dtype=torch.bfloat16), 48),
        (torch.empty((657, 2048), dtype=torch.bfloat16), 48),
        (torch.empty((1312, 2048), dtype=torch.bfloat16), 48),
        (torch.empty((1313, 2048), dtype=torch.bfloat16), 48),
        (torch.empty((497, 2048), dtype=torch.float32), 64),
    )
    for input_tensor, packed_n in cases:
        qweight = torch.empty((128, packed_n), dtype=torch.int32)
        gemm.paroquant_rotation_gemm_triton_prefill(
            input_tensor,
            qweight,
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
        )

    assert captured_configs == [
        (8, 2, True, False),
        (8, 4, True, True),
        (16, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (8, 1, False, False),
        (8, 1, True, True),
        (8, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (16, 1, False, False),
        (32, 1, False, False),
        (32, 1, False, False),
        (16, 1, False, False),
        (8, 2, True, False),
        (32, 2, True, False),
        (32, 2, True, False),
        (8, 2, True, False),
        (8, 4, True, True),
        (32, 4, True, True),
        (32, 4, True, True),
        (8, 4, True, True),
        (8, 1, True, False),
        (32, 1, True, False),
        (32, 1, True, False),
        (8, 1, True, False),
        (8, 1, True, True),
        (32, 1, True, True),
        (32, 1, True, True),
        (8, 1, True, True),
        (8, 1, True, False),
        (32, 1, True, False),
        (32, 1, True, False),
        (8, 1, True, False),
        (8, 1, True, True),
        (32, 1, True, True),
        (32, 1, True, True),
        (8, 1, True, True),
        (8, 1, True, False),
        (32, 1, True, False),
        (32, 1, True, False),
        (8, 1, True, False),
        (8, 1, True, True),
        (32, 1, True, True),
        (32, 1, True, True),
        (8, 1, True, True),
        (8, 4, True, False),
    ]


def test_paroquant_prefill_aligned_widths_route_measured_wave_bands(monkeypatch):
    """Keep the additional 128-aligned BM32 tiles inside their measured FP16 wave bands."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_configs = []

    def capture_config(*_args, **kwargs):
        captured_configs.append(
            (
                kwargs["block_size_m"],
                kwargs["num_warps"],
                kwargs["num_stages"],
                kwargs["loop_unroll_factor"],
                kwargs["explicit_fma"],
                kwargs["prefetch_first_partner"],
            )
        )
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    metadata = torch.empty((8, 0))
    cases = (
        (torch.empty((384, 2048), dtype=torch.float16), 80),
        (torch.empty((385, 2048), dtype=torch.float16), 80),
        (torch.empty((768, 2048), dtype=torch.float16), 80),
        (torch.empty((769, 2048), dtype=torch.float16), 80),
        (torch.empty((385, 2048), dtype=torch.bfloat16), 80),
        (torch.empty((768, 2048), dtype=torch.bfloat16), 80),
        (torch.empty((272, 2048), dtype=torch.float16), 112),
        (torch.empty((273, 2048), dtype=torch.float16), 112),
        (torch.empty((544, 2048), dtype=torch.float16), 112),
        (torch.empty((545, 2048), dtype=torch.float16), 112),
        (torch.empty((273, 2048), dtype=torch.bfloat16), 112),
        (torch.empty((544, 2048), dtype=torch.bfloat16), 112),
        (torch.empty((208, 2048), dtype=torch.float16), 144),
        (torch.empty((209, 2048), dtype=torch.float16), 144),
        (torch.empty((416, 2048), dtype=torch.float16), 144),
        (torch.empty((417, 2048), dtype=torch.float16), 144),
        (torch.empty((209, 2048), dtype=torch.bfloat16), 144),
        (torch.empty((416, 2048), dtype=torch.bfloat16), 144),
        (torch.empty((176, 2048), dtype=torch.float16), 176),
        (torch.empty((177, 2048), dtype=torch.float16), 176),
        (torch.empty((352, 2048), dtype=torch.float16), 176),
        (torch.empty((353, 2048), dtype=torch.float16), 176),
        (torch.empty((177, 2048), dtype=torch.bfloat16), 176),
        (torch.empty((352, 2048), dtype=torch.bfloat16), 176),
        (torch.empty((144, 2048), dtype=torch.float16), 208),
        (torch.empty((145, 2048), dtype=torch.float16), 208),
        (torch.empty((288, 2048), dtype=torch.float16), 208),
        (torch.empty((289, 2048), dtype=torch.float16), 208),
        (torch.empty((145, 2048), dtype=torch.bfloat16), 208),
        (torch.empty((288, 2048), dtype=torch.bfloat16), 208),
        (torch.empty((128, 2048), dtype=torch.float16), 240),
        (torch.empty((129, 2048), dtype=torch.float16), 240),
        (torch.empty((256, 2048), dtype=torch.float16), 240),
        (torch.empty((257, 2048), dtype=torch.float16), 240),
        (torch.empty((129, 2048), dtype=torch.bfloat16), 240),
        (torch.empty((256, 2048), dtype=torch.bfloat16), 240),
    )
    for input_tensor, packed_n in cases:
        qweight = torch.empty((128, packed_n), dtype=torch.int32)
        gemm.paroquant_rotation_gemm_triton_prefill(
            input_tensor,
            qweight,
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
        )

    assert captured_configs == [
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (32, 16, 1, 1, False, False),
        (32, 16, 1, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
        (16, 8, 2, 1, False, False),
    ]


def test_paroquant_prefill_wide_bm32_keeps_portable_launch_config(monkeypatch):
    """Use the measured schedule and extended widths only on their profiled SM inventory."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_configs = []

    def capture_config(*_args, **kwargs):
        captured_configs.append((kwargs["block_size_m"], kwargs["num_warps"], kwargs["num_stages"]))
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 108})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    metadata = torch.empty((8, 0))
    for out_features in (2048, 2560):
        gemm.paroquant_rotation_gemm_triton_prefill(
            torch.empty((97, 2048), dtype=torch.float16),
            torch.empty((128, out_features // 8), dtype=torch.int32),
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
            metadata,
        )

    assert captured_configs == [(32, 8, 2), (16, 8, 2)]


@pytest.mark.parametrize(
    ("sm_count", "rows", "expected"),
    [
        (124, 112, False),
        (124, 113, True),
        (124, 224, True),
        (124, 225, False),
        (124, 368, False),
        (124, 369, True),
        (124, 480, True),
        (124, 481, False),
        (124, 608, False),
        (124, 609, True),
        (124, 736, True),
        (124, 737, False),
        (124, 864, False),
        (124, 865, True),
        (124, 992, True),
        (124, 993, False),
        (108, 96, False),
        (108, 97, True),
        (108, 369, False),
        (80, 80, False),
        (80, 81, True),
        (80, 241, False),
    ],
)
def test_paroquant_prefill_wave_tile_uses_runtime_sm_count(monkeypatch, sm_count, rows, expected):
    """Derive the measured BM32 wave bands from the current device rather than its CUDA index."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    device_properties = type("DeviceProperties", (), {"multi_processor_count": sm_count})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)

    assert gemm._prefill_bm32_halves_cta_waves(torch.empty(rows, 2048), m=rows, n=2048) is expected


@pytest.mark.parametrize(
    ("sm_count", "out_features", "rows", "expected"),
    [
        (124, 512, 496, False),
        (124, 512, 497, True),
        (124, 512, 992, True),
        (124, 512, 993, False),
        (108, 512, 433, False),
        (80, 512, 321, False),
        (124, 256, 992, False),
        (124, 256, 993, True),
        (124, 256, 1984, True),
        (124, 256, 1985, False),
        (108, 256, 865, False),
        (80, 256, 641, False),
        (124, 128, 1984, False),
        (124, 128, 1985, True),
        (124, 128, 3968, True),
        (124, 128, 3969, False),
        (108, 128, 1729, False),
        (80, 128, 1281, False),
        (124, 384, 656, False),
        (124, 384, 657, True),
        (124, 384, 1312, True),
        (124, 384, 1313, False),
        (108, 384, 577, False),
        (80, 384, 433, False),
    ],
)
def test_paroquant_prefill_small_n_bm32_uses_only_measured_124sm_bands(
    monkeypatch,
    sm_count,
    out_features,
    rows,
    expected,
):
    """Keep the large-M small-N tiles inside their measured 124-SM one-wave bands."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    device_properties = type("DeviceProperties", (), {"multi_processor_count": sm_count})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)

    assert (
        gemm._prefill_small_n_bm32_collapses_waves(torch.empty(rows, 2048), m=rows, n=out_features) is expected
    )


@pytest.mark.parametrize(
    ("out_features", "rows", "expected"),
    [
        (1024, 240, False),
        (1024, 241, True),
        (1024, 480, True),
        (1024, 481, False),
        (1024, 737, False),
        (1536, 160, False),
        (1536, 161, True),
        (1536, 320, True),
        (1536, 321, False),
        (1536, 513, False),
        (640, 384, False),
        (640, 385, True),
        (640, 768, True),
        (640, 769, False),
        (640, 1185, False),
        (896, 272, False),
        (896, 273, True),
        (896, 544, True),
        (896, 545, False),
        (896, 849, False),
        (1152, 208, False),
        (1152, 209, True),
        (1152, 416, True),
        (1152, 417, False),
        (1152, 657, False),
        (1408, 176, False),
        (1408, 177, True),
        (1408, 352, True),
        (1408, 353, False),
        (1408, 529, False),
        (1664, 144, False),
        (1664, 145, True),
        (1664, 288, True),
        (1664, 289, False),
        (1664, 449, False),
        (1920, 128, False),
        (1920, 129, True),
        (1920, 256, True),
        (1920, 257, False),
        (1920, 385, False),
        (768, 320, False),
        (768, 321, True),
        (768, 640, True),
        (768, 641, False),
        (768, 993, False),
        (1280, 192, False),
        (1280, 193, True),
        (1280, 384, True),
        (1280, 385, False),
        (1280, 593, False),
        (1792, 128, False),
        (1792, 129, True),
        (1792, 256, True),
        (1792, 257, False),
        (1792, 417, False),
    ],
)
def test_paroquant_prefill_narrow_projection_uses_only_measured_one_wave_band(
    monkeypatch,
    out_features,
    rows,
    expected,
):
    """Keep unmeasured later narrow-projection BM32 wave bands disabled."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)

    assert gemm._prefill_bm32_halves_cta_waves(torch.empty(rows, 2048), m=rows, n=out_features) is expected


@pytest.mark.parametrize(
    ("out_features", "rows", "expected"),
    [
        (out_features, rows, expected)
        for out_features in range(2176, 4097, 128)
        for rows, expected in (
            (16 * (124 // (out_features // 128)), False),
            (16 * (124 // (out_features // 128)) + 1, True),
            (32 * (124 // (out_features // 128)), True),
            (32 * (124 // (out_features // 128)) + 1, False),
        )
    ],
)
def test_paroquant_prefill_extended_widths_use_only_measured_first_wave(
    monkeypatch,
    out_features,
    rows,
    expected,
):
    """Keep every extended width inside its measured first BM32 wave on the 124-SM target."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)

    assert gemm._prefill_bm32_halves_cta_waves(torch.empty(rows, 2048), m=rows, n=out_features) is expected


@pytest.mark.parametrize("out_features", range(2176, 4097, 128))
def test_paroquant_prefill_extended_widths_route_w16_on_measured_device(monkeypatch, out_features):
    """Route each retained extended first-wave band to BM32/W16/S1 only on the measured target."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_configs = []

    def capture_config(*_args, **kwargs):
        captured_configs.append((kwargs["block_size_m"], kwargs["num_warps"], kwargs["num_stages"]))
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    metadata = torch.empty((8, 0))
    rows = 16 * (124 // (out_features // 128)) + 1
    gemm.paroquant_rotation_gemm_triton_prefill(
        torch.empty((rows, 2048), dtype=torch.float16),
        torch.empty((128, out_features // 8), dtype=torch.int32),
        metadata,
        metadata,
        metadata,
        metadata,
        metadata,
        metadata,
    )

    assert captured_configs == [(32, 16, 1)]


def test_paroquant_decode_routes_measured_row_tiles_and_warps(monkeypatch):
    """Keep decode row tiles narrow only for the measured FP16 eight-rotation path."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_configs = []

    def capture_config(*_args, **kwargs):
        captured_configs.append(
            (
                kwargs["block_size_m"],
                kwargs["num_warps"],
                kwargs["loop_unroll_factor"],
                kwargs["explicit_fma"],
                kwargs["prefetch_first_partner"],
                kwargs["prefetch_packed_weight"],
            )
        )
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    qweight = torch.empty((128, 256), dtype=torch.int32)
    metadata = torch.empty((8, 0))
    cases = (
        (torch.empty((1, 2048), dtype=torch.float16), torch.empty((8, 0))),
        (torch.empty((2, 2048), dtype=torch.float16), torch.empty((8, 0))),
        (torch.empty((3, 2048), dtype=torch.float16), torch.empty((8, 0))),
        (torch.empty((8, 2048), dtype=torch.bfloat16), torch.empty((8, 0))),
        (torch.empty((1, 2048), dtype=torch.float16), torch.empty((1, 0))),
        (torch.empty((8, 2048), dtype=torch.bfloat16), torch.empty((1, 0))),
        (torch.empty((1, 512), dtype=torch.float16), torch.empty((8, 0))),
        (torch.empty((1, 256), dtype=torch.bfloat16), torch.empty((8, 0))),
        (torch.empty((1, 384), dtype=torch.bfloat16), torch.empty((8, 0))),
        (torch.empty((1, 896), dtype=torch.bfloat16), torch.empty((8, 0))),
        (torch.empty((1, 1024), dtype=torch.bfloat16), torch.empty((8, 0))),
    )
    for input_tensor, partner in cases:
        gemm.paroquant_rotation_gemm_triton_decode(
            input_tensor,
            qweight,
            metadata,
            metadata,
            partner,
            metadata,
            metadata,
            metadata,
        )

    assert captured_configs == [
        (2, 8, 1, True, False, False),
        (2, 8, 1, True, False, False),
        (4, 8, 1, True, False, False),
        (8, 8, 1, True, True, True),
        (8, 8, 1, False, False, False),
        (8, 8, 1, False, False, False),
        (2, 8, 1, False, False, False),
        (8, 8, 2, False, False, False),
        (8, 8, 1, False, True, False),
        (8, 8, 1, False, True, False),
        (8, 8, 1, True, True, True),
    ]


def test_paroquant_prefill_prefetches_weight_only_with_bf16_first_partner(monkeypatch):
    """Issue packed weights early only in the measured BF16 small-N latency-hiding regime."""
    from gptqmodel.quantization.paroquant.modules.triton import gemm

    captured_prefetches = []

    def capture_config(*_args, **kwargs):
        captured_prefetches.append(
            (kwargs["prefetch_first_partner"], kwargs["prefetch_packed_weight"])
        )
        return torch.empty(0)

    monkeypatch.setattr(gemm, "_paroquant_rotation_gemm_triton", capture_config)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    metadata = torch.empty((8, 0))
    cases = (
        (torch.empty((128, 2048), dtype=torch.bfloat16), 512, metadata),
        (torch.empty((128, 1024), dtype=torch.bfloat16), 512, metadata),
        (torch.empty((128, 896), dtype=torch.bfloat16), 512, metadata),
        (torch.empty((128, 2048), dtype=torch.bfloat16), 640, metadata),
        (torch.empty((128, 2048), dtype=torch.float16), 512, metadata),
        (torch.empty((128, 2048), dtype=torch.bfloat16), 512, torch.empty((1, 0))),
    )
    for input_tensor, out_features, partner in cases:
        gemm.paroquant_rotation_gemm_triton_prefill(
            input_tensor,
            torch.empty((128, out_features // 8), dtype=torch.int32),
            metadata,
            metadata,
            partner,
            metadata,
            metadata,
            metadata,
        )

    assert captured_prefetches == [
        (True, True),
        (True, True),
        (False, False),
        (False, False),
        (False, False),
        (False, False),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant kernel accuracy test")
def test_paroquant_cuda_matches_upstream_transformers_contract():
    """Compare the internal CUDA kernel path to an upstream-style contract.

    The official implementation rotates activations and then runs an AWQ-style
    packed matmul. This test reproduces that contract without importing upstream
    code and checks our fused CUDA path stays within a bounded numerical error.
    """
    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    torch.manual_seed(0)

    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.full((groups, out_features), 2 ** (bits - 1), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 0.75) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)
    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.25, 0.25)
    channel_scales.uniform_(0.7, 1.3)

    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(3, 7, in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = _upstream_transformers_contract_reference(
            x,
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bias=module.bias,
            pairs=module.pairs,
            theta=module.theta,
            channel_scales=module.channel_scales,
            bits=bits,
            group_size=group_size,
            out_features=out_features,
        )

        original_forward_dense = module._forward_dense
        try:
            module._forward_dense = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Expected ParoQuant CUDA kernel path, but dense fallback was used.")
            )
            actual = module(x)
        finally:
            module._forward_dense = original_forward_dense

    diff = (actual - expected).abs().float()
    assert diff.max().item() <= 0.25
    assert diff.mean().item() <= 0.03


def test_paroquant_backend_selection():
    """Guard user-facing backend selection for the default CUDA runtime."""
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=None,
        backend=BACKEND.PARO,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PARO,
        pack_dtype=torch.int32,
    )
    assert qlinear_cls is ParoLinear


def test_paroquant_triton_backend_mapping():
    """Guard registry lookup for the Triton ParoQuant runtime class."""
    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PARO, FORMAT.PAROQUANT)
        is ParoQuantTritonLinear
    )
    assert ParoQuantTritonLinear.SUPPORTS_FORMATS[FORMAT.PAROQUANT] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant Triton kernel parity test")
def test_paroquant_triton_matches_existing_cuda_kernel():
    """Guard Triton runtime accuracy against the established CUDA implementation."""
    pytest.importorskip("triton")

    bits = 4
    in_features = 128
    out_features = 128
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)

    baseline = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()
    candidate = ParoQuantTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda()

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    for module in (baseline, candidate):
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.bias.copy_(bias.cuda())
        module.pairs.copy_(pairs)
        module.theta.copy_(theta)
        module.channel_scales.copy_(channel_scales)
        module.post_init()
        module.eval()

    x = torch.randn(2, 8, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        rotated = apply_paroquant_rotation_reference(
            x,
            baseline.pairs,
            baseline.theta,
            scales=baseline.channel_scales,
            group_size=group_size,
        )
        dense_weight = dequantize_gemm(
            qweight=baseline.qweight,
            qzeros=baseline.qzeros,
            scales=baseline.scales,
            bits=bits,
            group_size=group_size,
        ).to(dtype=x.dtype)
        dense_reference = torch.matmul(rotated.reshape(-1, in_features), dense_weight).reshape(2, 8, out_features)
        dense_reference = dense_reference + baseline.bias

        baseline_out = baseline(x)
        candidate_out = candidate(x)

    baseline_max_abs = (baseline_out - dense_reference).abs().max().item()
    baseline_mean_abs = (baseline_out - dense_reference).abs().mean().item()
    candidate_max_abs = (candidate_out - dense_reference).abs().max().item()
    candidate_mean_abs = (candidate_out - dense_reference).abs().mean().item()

    assert candidate_max_abs <= baseline_max_abs + 0.1
    assert candidate_mean_abs <= baseline_mean_abs + 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant megakernel parity test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("krot", [1, 8])
@pytest.mark.parametrize(
    ("rows", "out_features", "kernel"),
    [
        (1, 192, paroquant_rotation_gemm_triton_decode),
        (8, 192, paroquant_rotation_gemm_triton_decode),
        (21, 256, paroquant_rotation_gemm_triton_prefill),
    ],
)
def test_paroquant_rotation_gemm_megakernel_matches_existing_cuda(dtype, krot, rows, out_features, kernel):
    """Compare one-launch decode/prefill paths with CUDA or the dense contract."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 256
    group_size = 128
    torch.manual_seed(123 + rows + krot)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=krot,
    ).to(device="cuda", dtype=dtype)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        device="cuda",
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.to(device="cuda", dtype=dtype))
    module.bias.copy_(bias.to(device="cuda", dtype=dtype))
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()

    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    partner, cos, sin = build_paroquant_rotation_lookup(
        module.pairs,
        module.theta,
        group_size=group_size,
    )
    if kernel is paroquant_rotation_gemm_triton_decode:
        partner_dtype = torch.int8 if dtype == torch.bfloat16 and krot == 8 else torch.int16
        partner = torch.remainder(partner, group_size).to(dtype=partner_dtype)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        actual = kernel(
            x,
            module.qweight,
            module.scales,
            module.qzeros,
            partner,
            cos,
            sin,
            module.channel_scales,
            module.bias,
        )
    torch.cuda.current_stream().wait_stream(stream)
    with torch.inference_mode():
        if out_features % 128 == 0:
            expected = module(x)
        else:
            rotated = module._rotate_inputs(x)
            expected = paroquant_gemm_triton_decode(
                rotated,
                module.qweight,
                module.scales,
                module.qzeros,
            )
            expected = expected + module.bias

    diff = (actual - expected).abs().float()
    assert actual.shape == expected.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert diff.max().item() <= (2.0 if dtype == torch.bfloat16 else 0.5)
    assert diff.mean().item() <= 0.02


def _make_combined_awq_dispatch_module(*, dtype: torch.dtype, bias: bool) -> ParoLinear:
    """Build a compact non-identity module for combined native-dispatch checks."""
    bits = 4
    in_features = 256
    out_features = 256
    group_size = 128
    krot = 8
    torch.manual_seed(601 + int(dtype == torch.bfloat16) + int(bias))
    qweight, qzeros, scales, bias_values = _make_packed_buffers(
        bits,
        in_features,
        out_features,
        group_size,
    )
    module = ParoLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        register_buffers=True,
        krot=krot,
    ).to(device="cuda", dtype=dtype)
    pairs, _ = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=17,
        device=torch.device("cuda"),
    )
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.to(device="cuda", dtype=dtype))
    if bias:
        module.bias.copy_(bias_values.to(device="cuda", dtype=dtype))
    module.pairs.copy_(pairs)
    module.theta.uniform_(-0.2, 0.2)
    module.channel_scales.uniform_(0.75, 1.25)
    module.post_init()
    return module.eval()


def _require_combined_awq_dtype_support(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BFloat16 combined ParoQuant AWQ dispatch requires compute capability >= 8.0.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for combined ParoQuant AWQ dispatch test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_paroquant_combined_awq_dispatch_matches_separate_ops(dtype, bias, monkeypatch):
    """The one-dispatch path must preserve the established rotation plus AWQ result exactly."""
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA extension entrypoint unavailable.")
    _require_combined_awq_dtype_support(dtype)
    module = _make_combined_awq_dispatch_module(dtype=dtype, bias=bias)
    x = torch.randn((1, 33, module.in_features), device="cuda", dtype=dtype)

    with torch.inference_mode():
        module.paroquant_cuda_awq_fused_dispatch_enabled = False
        expected = module(x)
        module.paroquant_cuda_awq_fused_dispatch_enabled = True
        if bias:
            monkeypatch.setenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE_BIAS", "1")
            separate_bias = module(x)
            monkeypatch.delenv("GPTQMODEL_AWQ_DISABLE_FUSED_SPLITK_REDUCE_BIAS")
        actual = module(x)
        repeated = module(x)

    if bias:
        torch.testing.assert_close(separate_bias, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for combined ParoQuant AWQ graph test")
def test_paroquant_combined_awq_dispatch_preserves_graph_and_stream_ownership():
    """Graph replay and concurrent streams must receive independent, exact outputs."""
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA extension entrypoint unavailable.")
    _require_combined_awq_dtype_support(torch.bfloat16)
    module = _make_combined_awq_dispatch_module(dtype=torch.bfloat16, bias=True)
    x = torch.randn((1, 128, module.in_features), device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        module.paroquant_cuda_awq_fused_dispatch_enabled = False
        expected = module(x)
        module.paroquant_cuda_awq_fused_dispatch_enabled = True
        module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        for _ in range(3):
            graph.replay()

        first_stream = torch.cuda.Stream()
        second_stream = torch.cuda.Stream()
        with torch.cuda.stream(first_stream):
            first = module(x)
        with torch.cuda.stream(second_stream):
            second = module(x)
        torch.cuda.current_stream().wait_stream(first_stream)
        torch.cuda.current_stream().wait_stream(second_stream)

    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert len({captured.data_ptr(), first.data_ptr(), second.data_ptr()}) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for combined ParoQuant AWQ fallback test")
def test_paroquant_combined_awq_dispatch_failure_keeps_separate_fallback(monkeypatch):
    """An optional combined-op failure must preserve the established separate launch path."""
    if not awq_runtime_available():
        pytest.skip("AWQ CUDA extension entrypoint unavailable.")
    module = _make_combined_awq_dispatch_module(dtype=torch.float16, bias=True)
    x = torch.randn((1, 33, module.in_features), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module.paroquant_cuda_awq_fused_dispatch_enabled = False
        expected = module(x)
        module.paroquant_cuda_awq_fused_dispatch_enabled = True
        monkeypatch.setattr(
            "gptqmodel.nn_modules.qlinear.paroquant.apply_paroquant_rotation_awq",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced combined-op failure")),
        )
        actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant wide-prefill tile test")
@pytest.mark.parametrize(
    ("rows", "out_features"),
    [
        (113, 2048),
        (128, 2048),
        (224, 2048),
        (369, 2048),
        (480, 2048),
        (609, 2048),
        (736, 2048),
        (865, 2048),
        (992, 2048),
        (241, 1024),
        (480, 1024),
        (161, 1536),
        (320, 1536),
        (321, 768),
        (640, 768),
        (193, 1280),
        (384, 1280),
        (129, 1792),
        (256, 1792),
        (385, 640),
        (768, 640),
        (273, 896),
        (544, 896),
        (209, 1152),
        (416, 1152),
        (177, 1408),
        (352, 1408),
        (145, 1664),
        (288, 1664),
        (129, 1920),
        (256, 1920),
        *[
            (rows, out_features)
            for out_features in range(2176, 4097, 128)
            for rows in (
                16 * (124 // (out_features // 128)) + 1,
                32 * (124 // (out_features // 128)),
            )
        ],
    ],
)
def test_paroquant_rotation_gemm_megakernel_wide_prefill_matches_bm16(rows, out_features):
    """Keep the retained FP16 BM32 wave route bit-identical to the established BM16 tile."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 2048
    group_size = 128
    krot = 8
    dtype = torch.float16
    torch.manual_seed(321)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        device="cuda",
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=group_size)
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    bias = bias.cuda()

    with torch.inference_mode():
        expected = _paroquant_rotation_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            block_size_m=16,
            block_size_n=128,
            num_warps=8,
            num_stages=2,
        )
        actual = paroquant_rotation_gemm_triton_prefill(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant small-N tile test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("out_features", "rows"),
    [
        (512, 497),
        (512, 992),
        (256, 993),
        (256, 1984),
        (128, 1985),
        (128, 3968),
        (384, 657),
        (384, 1312),
    ],
)
def test_paroquant_rotation_gemm_megakernel_small_n_prefill_matches_bm8(dtype, out_features, rows):
    """Keep the measured 124-SM small-N BM32 routes bit-identical to the established BM8 tile."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")
    if torch.cuda.get_device_properties("cuda").multi_processor_count != 124:
        pytest.skip("The retained small-N BM32 routes are measured only on the 124-SM target.")

    bits = 4
    in_features = 2048
    group_size = 128
    krot = 8
    torch.manual_seed(512 + rows)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    pairs, _ = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=57,
        device=torch.device("cuda"),
    )
    theta = torch.empty((krot, in_features // 2), device="cuda", dtype=dtype).uniform_(-0.2, 0.2)
    channel_scales = torch.empty((1, in_features), device="cuda", dtype=dtype).uniform_(0.75, 1.25)
    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=group_size)
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.to(device="cuda", dtype=dtype)
    bias = bias.to(device="cuda", dtype=dtype)
    if out_features == 512:
        loop_unroll_factor = 2 if dtype == torch.float16 else 4
    else:
        loop_unroll_factor = 1
    prefetch_first_partner = dtype == torch.bfloat16

    with torch.inference_mode():
        expected = _paroquant_rotation_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            block_size_m=8,
            block_size_n=128,
            num_warps=8,
            num_stages=2,
            loop_unroll_factor=loop_unroll_factor,
            explicit_fma=True,
            prefetch_first_partner=prefetch_first_partner,
        )
        actual = paroquant_rotation_gemm_triton_prefill(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant prefetch parity test")
@pytest.mark.parametrize(
    ("rows", "out_features", "decode", "baseline_unroll", "candidate_unroll"),
    [
        (1, 2048, True, 2, 1),
        (128, 512, False, 4, 4),
    ],
)
def test_paroquant_first_partner_prefetch_matches_retained_schedule(
    rows,
    out_features,
    decode,
    baseline_unroll,
    candidate_unroll,
):
    """Keep the BF16 latency-hiding prefetches bit-identical for randomized pair schedules."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 2048
    group_size = 128
    krot = 8
    dtype = torch.bfloat16
    torch.manual_seed(913 + rows)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    pairs, _ = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=27,
        device=torch.device("cuda"),
    )
    theta = torch.empty((krot, in_features // 2), device="cuda", dtype=dtype).uniform_(-0.2, 0.2)
    channel_scales = torch.empty((1, in_features), device="cuda", dtype=dtype).uniform_(0.75, 1.25)
    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=group_size)
    if decode:
        partner = torch.remainder(partner, group_size).to(dtype=torch.int8)

    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.to(device="cuda", dtype=dtype)
    bias = bias.to(device="cuda", dtype=dtype)
    with torch.inference_mode():
        baseline = _paroquant_rotation_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            block_size_m=8,
            block_size_n=128,
            num_warps=8,
            num_stages=2,
            loop_unroll_factor=baseline_unroll,
            explicit_fma=True,
            prefetch_first_partner=False,
            prefetch_packed_weight=False,
        )
        candidate = _paroquant_rotation_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            block_size_m=8,
            block_size_n=128,
            num_warps=8,
            num_stages=2,
            loop_unroll_factor=candidate_unroll,
            explicit_fma=True,
            prefetch_first_partner=True,
            prefetch_packed_weight=True,
        )

    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "rows", "out_features", "expected"),
    [
        (torch.float16, 1, 512, (8, 8)),
        (torch.float16, 2, 512, (8, 8)),
        (torch.float16, 8, 512, (8, 8)),
        (torch.float16, 1, 2048, (2, 4)),
        (torch.float16, 1, 8192, (2, 4)),
        (torch.float16, 2, 8192, (2, 4)),
        (torch.float16, 3, 8192, (4, 4)),
        (torch.float16, 4, 8192, (4, 4)),
        (torch.float16, 5, 8192, (8, 8)),
        (torch.float16, 8, 8192, (8, 8)),
        (torch.float16, 2, 2048, (4, 4)),
        (torch.float16, 3, 2048, (4, 4)),
        (torch.float16, 4, 2048, (4, 4)),
        (torch.float16, 5, 2048, (8, 4)),
        (torch.float16, 6, 2048, (8, 4)),
        (torch.float16, 7, 2048, (8, 4)),
        (torch.float16, 8, 2048, (8, 8)),
        (torch.float16, 9, 512, (32, 4)),
        (torch.float16, 129, 1920, (32, 4)),
        (torch.float16, 192, 2560, (32, 4)),
        (torch.float16, 160, 3072, (32, 4)),
        (torch.float16, 96, 4096, (32, 4)),
        (torch.bfloat16, 1, 2048, (8, 8)),
        (torch.bfloat16, 1, 8192, (4, 4)),
    ],
)
def test_paroquant_splitk_launch_config_uses_only_measured_dtype_shapes(dtype, rows, out_features, expected):
    assert _paroquant_splitk_launch_config(dtype, rows=rows, out_features=out_features) == expected


@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "split_k", "expected"),
    [
        (1, 4096, 1024, 32, (4, 4)),
        (1, 4096, 4096, 32, (4, 4)),
        (1, 4096, 12288, 32, (4, 4)),
        (8, 4096, 1024, 32, (8, 8)),
        (8, 4096, 4096, 32, (8, 8)),
        (8, 4096, 12288, 32, (8, 8)),
        (1, 12288, 4096, 96, (1, 4)),
        (8, 12288, 4096, 32, (8, 8)),
        (2, 4096, 1024, 32, (8, 8)),
        (2, 4096, 4096, 32, (4, 4)),
        (4, 4096, 4096, 32, (4, 4)),
        (4, 4096, 12288, 32, (8, 8)),
        (16, 4096, 4096, 32, (8, 8)),
        (32, 4096, 1024, 32, (8, 8)),
        (32, 4096, 4096, 32, (32, 8)),
        (1, 4096, 4096, 16, (8, 8)),
    ],
)
def test_paroquant_large_k_splitk_launch_config_uses_only_measured_qwen_shapes(
    rows,
    in_features,
    out_features,
    split_k,
    expected,
):
    assert (
        _paroquant_splitk_launch_config(
            torch.bfloat16,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            split_k=split_k,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dtype", "rows", "in_features", "out_features", "split_k", "expected"),
    [
        (torch.float16, 1, 2048, 8192, 16, (128, 1, 2, 136, True, False)),
        (torch.float16, 2, 2048, 8192, 16, (128, 1, 2, 168, True, False)),
        (torch.float16, 3, 2048, 8192, 16, (128, 1, 2, 144, True, False)),
        (torch.float16, 4, 2048, 8192, 16, (128, 1, 2, 120, True, False)),
        (torch.float16, 5, 2048, 8192, 16, (128, 1, 2, 76, True, False)),
        (torch.float16, 6, 2048, 8192, 16, (128, 1, 2, 76, True, True)),
        (torch.float16, 7, 2048, 8192, 16, (128, 1, 2, 76, True, True)),
        (torch.float16, 8, 2048, 8192, 16, (128, 1, 2, 76, True, True)),
        (torch.float16, 8, 2048, 8192, 8, (128, 2, 1, None, False, False)),
        (torch.float16, 8, 4096, 8192, 16, (128, 2, 1, None, False, False)),
        (torch.float16, 8, 2048, 2048, 16, (128, 2, 1, None, False, False)),
        (torch.float16, 129, 2048, 512, 16, (128, 1, 1, None, False, False)),
        (torch.float16, 129, 2048, 1920, 16, (128, 1, 1, None, False, False)),
        (torch.float16, 129, 4096, 1920, 16, (128, 2, 1, None, False, False)),
        (torch.float16, 129, 2048, 1920, 8, (128, 2, 1, None, False, False)),
        (torch.bfloat16, 129, 2048, 1920, 16, (128, 2, 1, None, False, False)),
        (torch.bfloat16, 8, 2048, 8192, 16, (128, 2, 1, None, False, False)),
        (torch.bfloat16, 1, 4096, 12288, 32, (128, 2, 2, 128, True, False)),
        (torch.bfloat16, 8, 4096, 12288, 32, (128, 2, 2, None, False, False)),
        (torch.bfloat16, 1, 4096, 4096, 32, (128, 2, 1, 128, False, False)),
        (torch.bfloat16, 2, 4096, 1024, 32, (128, 2, 1, None, False, False)),
        (torch.bfloat16, 2, 4096, 4096, 32, (128, 2, 2, 144, True, True)),
        (torch.bfloat16, 4, 4096, 4096, 32, (128, 2, 1, None, False, False)),
        (torch.bfloat16, 32, 4096, 4096, 32, (128, 1, 2, 128, True, True)),
        (torch.bfloat16, 1, 12288, 4096, 96, (128, 2, 2, 128, True, False)),
        (torch.bfloat16, 8, 12288, 4096, 32, (128, 2, 1, None, False, False)),
    ],
)
def test_paroquant_splitk_output_config_uses_only_measured_fp16_shapes(
    dtype,
    rows,
    in_features,
    out_features,
    split_k,
    expected,
):
    assert (
        _paroquant_splitk_output_config(
            dtype,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            split_k=split_k,
        )
        == expected
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    ("in_features", "out_features", "expected"),
    [
        (4096, 1024, 32),
        (4096, 4096, 32),
        (4096, 12288, 32),
        (12288, 4096, 32),
    ],
)
def test_paroquant_large_k_splitk_factor_covers_target_qwen_rows(rows, in_features, out_features, expected):
    if rows == 1 and in_features == 12288:
        expected = 96
    assert (
        ParoQuantTritonLinear._megakernel_large_k_decode_splitk_factor(
            torch.bfloat16,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dtype", "rows"),
    [
        (torch.float16, 1),
        (torch.bfloat16, 3),
        (torch.bfloat16, 64),
    ],
)
def test_paroquant_large_k_splitk_factor_rejects_unmeasured_rows_and_dtype(dtype, rows):
    assert (
        ParoQuantTritonLinear._megakernel_large_k_decode_splitk_factor(
            dtype,
            rows=rows,
            in_features=4096,
            out_features=4096,
        )
        is None
    )


@pytest.mark.parametrize(
    ("rows", "out_features", "expected"),
    [
        (1, 512, True),
        (1, 2048, True),
        (1, 8192, True),
        (2, 512, True),
        (2, 2048, True),
        (2, 8192, True),
        (3, 512, True),
        (3, 2048, True),
        (3, 8192, True),
        (4, 512, True),
        (4, 2048, True),
        (4, 8192, True),
        (5, 512, True),
        (5, 2048, True),
        (5, 8192, True),
        (6, 512, True),
        (6, 2048, True),
        (6, 8192, True),
        (7, 512, True),
        (7, 2048, True),
        (7, 8192, True),
        (8, 512, True),
        (8, 2048, True),
        (8, 8192, True),
        (1, 1024, False),
        (8, 4096, False),
        (9, 512, False),
        (9, 2048, False),
        (9, 8192, False),
    ],
)
def test_paroquant_fp16_splitk_gate_uses_only_measured_shapes(rows, out_features, expected):
    assert (
        ParoQuantTritonLinear._megakernel_fp16_splitk_shape(
            rows=rows,
            out_features=out_features,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("rows", "out_features", "expected"),
    [
        (8, 512, False),
        (9, 512, True),
        (992, 512, True),
        (993, 512, False),
        (9, 1920, True),
        (256, 1920, True),
        (257, 1920, False),
        (9, 2048, True),
        (256, 2048, True),
        (96, 2560, False),
        (97, 2560, True),
        (192, 2560, True),
        (193, 2560, False),
        (80, 3072, False),
        (81, 3072, True),
        (160, 3072, True),
        (161, 3072, False),
        (48, 4096, False),
        (49, 4096, True),
        (96, 4096, True),
        (97, 4096, False),
        (129, 1024, False),
    ],
)
def test_paroquant_fp16_prefill_splitk_gate_uses_only_measured_bands(rows, out_features, expected):
    assert (
        ParoQuantTritonLinear._megakernel_fp16_prefill_splitk_shape(
            rows=rows,
            out_features=out_features,
        )
        is expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K parity test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "out_features"),
    [
        *[(rows, 512) for rows in range(1, 9)],
        *[(rows, 2048) for rows in range(1, 9)],
        *[(rows, 8192) for rows in range(1, 9)],
    ],
)
def test_paroquant_splitk_decode_matches_standard_megakernel(dtype, rows, out_features):
    """Bound split-K reduction drift and require deterministic scratch reuse."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 2048
    group_size = 128
    krot = 8
    torch.manual_seed(1600 + rows + out_features)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    pairs, _ = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=91,
        device=torch.device("cuda"),
    )
    theta = torch.empty((krot, in_features // 2), device="cuda", dtype=dtype).uniform_(-0.2, 0.2)
    channel_scales = torch.empty((1, in_features), device="cuda", dtype=dtype).uniform_(0.75, 1.25)
    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=group_size)
    partner_dtype = torch.int8 if dtype == torch.bfloat16 else torch.int16
    partner = torch.remainder(partner, group_size).to(dtype=partner_dtype)
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.to(device="cuda", dtype=dtype)
    bias = bias.to(device="cuda", dtype=dtype)
    split_k = in_features // group_size
    block_size_m, _ = _paroquant_splitk_launch_config(dtype, rows=rows, out_features=out_features)
    num_tiles = (out_features + 127) // 128
    partials = torch.empty(num_tiles * split_k * block_size_m * 128, device="cuda", dtype=torch.float32)
    counters = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)

    with torch.inference_mode():
        expected = paroquant_rotation_gemm_triton_decode(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
        )
        actual = _paroquant_rotation_gemm_splitk_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            partials,
            counters,
            split_k=split_k,
        )
        repeated = _paroquant_rotation_gemm_splitk_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            partials,
            counters,
            split_k=split_k,
        )

    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=2.0)
    assert (actual - expected).abs().float().mean().item() <= 0.003
    assert counters.count_nonzero().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K dense test")
@pytest.mark.parametrize("rows", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paroquant_splitk_decode_preserves_dense_error_envelope(dtype, rows):
    """Keep split-K error within the standard mega-kernel's dense-reference envelope."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 2048
    out_features = 2048
    group_size = 128
    krot = 8
    torch.manual_seed(2048 + rows)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        device="cuda",
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    partner, cos, sin = build_paroquant_rotation_lookup(pairs, theta, group_size=group_size)
    partner_dtype = torch.int8 if dtype == torch.bfloat16 else torch.int16
    partner = torch.remainder(partner, group_size).to(dtype=partner_dtype)
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.to(device="cuda", dtype=dtype)
    bias = bias.to(device="cuda", dtype=dtype)
    block_size_m, _ = _paroquant_splitk_launch_config(dtype, rows=rows, out_features=out_features)
    num_tiles = ((rows + block_size_m - 1) // block_size_m) * (out_features // 128)
    partials = torch.empty(num_tiles * 16 * block_size_m * 128, device="cuda", dtype=torch.float32)
    counters = torch.zeros(num_tiles, device="cuda", dtype=torch.int32)

    with torch.inference_mode():
        rotated = apply_paroquant_rotation_reference(
            x,
            pairs,
            theta,
            scales=channel_scales,
            group_size=group_size,
        )
        dense_weight = dequantize_gemm(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            bits=bits,
            group_size=group_size,
        ).to(dtype=dtype)
        dense = torch.matmul(rotated, dense_weight) + bias
        standard = paroquant_rotation_gemm_triton_decode(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
        )
        split = _paroquant_rotation_gemm_splitk_triton(
            x,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            partials,
            counters,
            split_k=16,
        )

    standard_error = (standard - dense).abs().float()
    split_error = (split - dense).abs().float()
    assert split_error.max().item() <= standard_error.max().item() + 2.0
    assert split_error.mean().item() <= standard_error.mean().item() + 0.003


def _make_splitk_triton_module(
    *,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    out_features: int = 2048,
) -> ParoQuantTritonLinear:
    """Build a production-shaped module for split-K integration tests."""
    bits = 4
    in_features = 2048
    group_size = 128
    krot = 8
    torch.manual_seed(seed)
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    module = ParoQuantTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=krot,
    ).to(device="cuda", dtype=dtype).eval()
    pairs, _ = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=seed,
        device=torch.device("cuda"),
    )
    theta = torch.empty((krot, in_features // 2), device="cuda", dtype=dtype).uniform_(-0.2, 0.2)
    channel_scales = torch.empty((1, in_features), device="cuda", dtype=dtype).uniform_(0.75, 1.25)
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.to(device="cuda", dtype=dtype))
    module.bias.copy_(bias.to(device="cuda", dtype=dtype))
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.paroquant_triton_autotune_enabled = False
    return module


def _make_large_k_splitk_modules(
    *,
    seed: int,
    in_features: int,
    out_features: int,
) -> tuple[ParoLinear, ParoQuantTritonLinear]:
    """Build reference and candidate modules without a dense K-by-N integer temporary."""
    bits = 4
    group_size = 128
    krot = 8
    dtype = torch.bfloat16
    torch.manual_seed(seed)
    groups = in_features // group_size
    qweight = torch.randint(-(2**31), 2**31 - 1, (in_features, out_features // 8), dtype=torch.int32)
    qzeros = torch.randint(-(2**31), 2**31 - 1, (groups, out_features // 8), dtype=torch.int32)
    scales = ((torch.rand(groups, out_features) * 0.04) + 0.01).to(dtype)
    bias = (torch.randn(out_features) * 0.1).to(dtype)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    modules = []
    for module_cls in (ParoLinear, ParoQuantTritonLinear):
        module = module_cls(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=True,
            register_buffers=True,
            krot=krot,
        ).to(device="cuda", dtype=dtype)
        module.qweight.copy_(qweight.cuda())
        module.qzeros.copy_(qzeros.cuda())
        module.scales.copy_(scales.cuda())
        module.bias.copy_(bias.cuda())
        module.pairs.copy_(pairs.cuda())
        module.theta.copy_(theta.cuda())
        module.channel_scales.copy_(channel_scales.cuda())
        module.post_init()
        module.eval()
        modules.append(module)
    candidate = modules[1]
    candidate.paroquant_triton_autotune_enabled = False
    return modules[0], candidate


def test_paroquant_cached_cuda_awq_plan_skips_general_megakernel_routing(monkeypatch):
    """Submit an already-selected fused CUDA fallback before re-entering the general router."""
    module = object.__new__(ParoQuantTritonLinear)
    torch.nn.Module.__init__(module)
    module.in_features = 4096
    module.out_features = 12288
    module.adapter = None
    x = torch.randn((1, 16, module.in_features))
    cache_key = ("decode", 16, module.in_features, module.out_features, x.dtype, x.device)
    module._plan_cache = {cache_key: "cuda_awq"}
    expected = torch.randn((16, module.out_features))
    fused_inputs = []

    def fused(input):
        fused_inputs.append(input)
        return expected

    monkeypatch.setattr(module, "_forward_cuda_awq_fused", fused)

    actual = module.forward(x)

    assert fused_inputs == [x]
    assert actual.shape == (1, 16, module.out_features)
    assert actual._base is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant large-K split-K test")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features"),
    [
        *[(rows, 4096, 1024) for rows in (1, 2, 4, 8, 16, 32)],
        *[(rows, 4096, 4096) for rows in (1, 2, 4, 8, 16, 32)],
        *[(rows, 4096, 12288) for rows in (1, 2, 4, 8, 16, 32)],
        *[(rows, 12288, 4096) for rows in (1, 2, 4, 8, 16, 32)],
    ],
)
def test_paroquant_large_k_splitk_qwen_decode_preserves_reference_accuracy(rows, in_features, out_features):
    """Run every retained large-K schedule through production before timing it."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")
    if torch.cuda.get_device_properties(0).multi_processor_count != 124:
        pytest.skip("ParoQuant large-K split-K is measured only on the 124-SM target.")

    reference, candidate = _make_large_k_splitk_modules(
        seed=4096 + rows + out_features,
        in_features=in_features,
        out_features=out_features,
    )
    torch.manual_seed(12288 + rows + out_features)
    x = torch.randn((1, rows, in_features), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference(x)
        actual = candidate(x)
        repeated = candidate(x)

    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)
    difference = (actual - expected).abs().float()
    reference_mean_abs = expected.abs().float().mean().item()
    assert difference.mean().item() / reference_mean_abs <= 0.005
    assert set(candidate._plan_cache.values()) == {"decode_megakernel"}

    split_k = candidate._megakernel_decode_splitk_factor(x, rows=rows)
    assert split_k is not None
    block_m, _ = _paroquant_splitk_launch_config(
        x.dtype,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        split_k=split_k,
    )
    block_n, _, _, _, _, _ = _paroquant_splitk_output_config(
        x.dtype,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        split_k=split_k,
    )
    num_tiles = ((rows + block_m - 1) // block_m) * ((out_features + block_n - 1) // block_n)
    partials, counters = next(iter(candidate._megakernel_splitk_scratch.values()))
    assert partials.numel() == num_tiles * split_k * block_m * block_n
    assert counters.numel() == num_tiles
    assert counters.count_nonzero().item() == 0
    compiled_key = (x.device, rows, out_features, x.dtype)
    compiled_kernel = candidate._megakernel_splitk_compiled.get(compiled_key)
    assert compiled_kernel is not None and compiled_kernel is not False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant large-K graph test")
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [
        (4096, 12288),
        (12288, 4096),
    ],
)
def test_paroquant_large_k_splitk_qwen_decode_replays_from_private_cuda_graph_scratch(
    in_features,
    out_features,
):
    """Replay the measured two-output split-32 and split-96 schedules through their compiled launchers."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")
    if torch.cuda.get_device_properties(0).multi_processor_count != 124:
        pytest.skip("ParoQuant large-K split-K is measured only on the 124-SM target.")

    _, module = _make_large_k_splitk_modules(
        seed=8192 + in_features + out_features,
        in_features=in_features,
        out_features=out_features,
    )
    torch.manual_seed(16384 + in_features + out_features)
    x = torch.randn((1, 1, in_features), device="cuda", dtype=torch.bfloat16)
    scratch_results = []
    original_scratch = module._megakernel_decode_splitk_scratch

    def record_scratch(x_flat, **kwargs):
        scratch = original_scratch(x_flat, **kwargs)
        scratch_results.append(scratch)
        return scratch

    module._megakernel_decode_splitk_scratch = record_scratch
    with torch.inference_mode():
        expected = module(x)
        scratch_results.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        for _ in range(3):
            graph.replay()
        actual = captured.clone()

    assert len(scratch_results) == 1
    assert scratch_results[0] is not None
    eager_partials, eager_counters = next(iter(module._megakernel_splitk_scratch.values()))
    graph_partials, graph_counters, _ = scratch_results[0]
    assert graph_partials.data_ptr() != eager_partials.data_ptr()
    assert graph_counters.data_ptr() != eager_counters.data_ptr()
    assert graph_counters.count_nonzero().item() == 0
    assert eager_counters.count_nonzero().item() == 0
    assert actual.shape == (1, 1, out_features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K prefill test")
def test_paroquant_splitk_fp16_prefill_uses_compact_production_path_and_preserves_dense_error():
    """Run measured prefill split-K through production with compact scratch and a dense baseline."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    rows = 129
    module = _make_splitk_triton_module(seed=2529, dtype=torch.float16, out_features=512)
    x = torch.randn(1, rows, 2048, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        rotated = apply_paroquant_rotation_reference(
            x.reshape(rows, 2048),
            module.pairs,
            module.theta,
            scales=module.channel_scales,
            group_size=module.group_size,
        )
        dense_weight = dequantize_gemm(
            qweight=module.qweight,
            qzeros=module.qzeros,
            scales=module.scales,
            bits=module.bits,
            group_size=module.group_size,
        ).to(dtype=torch.float16)
        dense = torch.matmul(rotated, dense_weight) + module.bias

        module.paroquant_triton_megakernel_prefill_splitk_enabled = False
        standard = module(x).reshape(rows, module.out_features)
        module.paroquant_triton_megakernel_prefill_splitk_enabled = True
        split = module(x).reshape(rows, module.out_features)
        repeated = module(x).reshape(rows, module.out_features)

    torch.testing.assert_close(repeated, split, rtol=0, atol=0)
    torch.testing.assert_close(split, standard, rtol=0.01, atol=2.0)
    assert (split - standard).abs().float().mean().item() <= 0.003
    standard_error = (standard - dense).abs().float()
    split_error = (split - dense).abs().float()
    assert split_error.max().item() <= standard_error.max().item() + 2.0
    assert split_error.mean().item() <= standard_error.mean().item() + 0.003

    compiled_key = (x.device, rows, module.out_features, x.dtype)
    compiled_kernel = module._megakernel_splitk_compiled.get(compiled_key)
    assert compiled_kernel is not None and compiled_kernel is not False
    assert module._megakernel_splitk_prepared == {}
    partials, counters = next(iter(module._megakernel_splitk_scratch.values()))
    num_tiles = ((rows + 31) // 32) * (module.out_features // 128)
    assert partials.numel() == num_tiles * 16 * 32 * 128
    assert partials.numel() < num_tiles * 16 * rows * 128
    assert counters.numel() == num_tiles
    assert counters.count_nonzero().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K prefill graph test")
def test_paroquant_splitk_fp16_prefill_replays_from_private_cuda_graph_scratch():
    """Keep warmed prefill split-K deterministic when capture owns a private scratch allocation."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(seed=2530, dtype=torch.float16, out_features=512)
    x = torch.randn(1, 33, 2048, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = module(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        for _ in range(3):
            graph.replay()
        actual = captured.clone()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K row test")
@pytest.mark.parametrize("rows", [3, 5, 6, 7])
def test_paroquant_splitk_fp16_irregular_rows_use_compiled_path(rows):
    """Run every newly measured FP16 row count through the warmed production path."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(seed=2400 + rows, dtype=torch.float16)
    x = torch.randn(1, rows, 2048, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module.paroquant_triton_megakernel_decode_splitk_enabled = False
        standard = module(x)
        module.paroquant_triton_megakernel_decode_splitk_enabled = True
        split = module(x)
        repeated = module(x)

    torch.testing.assert_close(repeated, split, rtol=0, atol=0)
    torch.testing.assert_close(split, standard, rtol=0.01, atol=2.0)
    assert (split - standard).abs().float().mean().item() <= 0.003
    compiled_key = (x.device, rows, module.out_features, x.dtype)
    compiled_kernel = module._megakernel_splitk_compiled.get(compiled_key)
    assert compiled_kernel is not None and compiled_kernel is not False
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K width test")
@pytest.mark.parametrize(
    ("rows", "out_features"),
    [
        (2, 512),
        (8, 512),
        (1, 8192),
        (2, 8192),
        (3, 8192),
        (4, 8192),
        (5, 8192),
        (6, 8192),
        (7, 8192),
        (8, 8192),
    ],
)
def test_paroquant_splitk_fp16_new_width_rows_use_compiled_path(rows, out_features):
    """Run representative newly measured FP16 widths through the warmed production path."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(
        seed=2450 + rows + out_features,
        dtype=torch.float16,
        out_features=out_features,
    )
    x = torch.randn(1, rows, 2048, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module.paroquant_triton_megakernel_decode_splitk_enabled = False
        standard = module(x)
        module.paroquant_triton_megakernel_decode_splitk_enabled = True
        split = module(x)
        repeated = module(x)

    torch.testing.assert_close(repeated, split, rtol=0, atol=0)
    torch.testing.assert_close(split, standard, rtol=0.01, atol=2.0)
    assert (split - standard).abs().float().mean().item() <= 0.003
    compiled_key = (x.device, rows, module.out_features, x.dtype)
    compiled_kernel = module._megakernel_splitk_compiled.get(compiled_key)
    assert compiled_kernel is not None and compiled_kernel is not False
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K dtype test")
@pytest.mark.parametrize(("rows", "out_features"), [(1, 2048), (8, 8192)])
def test_paroquant_splitk_fp16_compiled_path_is_dtype_isolated(rows, out_features):
    """Keep warmed FP16/BF16 launchers separate while reusing dtype-agnostic scratch."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(
        seed=2501 + rows + out_features,
        dtype=torch.float16,
        out_features=out_features,
    )
    inputs = {
        torch.float16: torch.randn(rows, 2048, device="cuda", dtype=torch.float16),
        torch.bfloat16: torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16),
    }

    with torch.inference_mode():
        for dtype, x in inputs.items():
            module.paroquant_triton_megakernel_decode_splitk_enabled = False
            standard = module(x)
            module.paroquant_triton_megakernel_decode_splitk_enabled = True
            split = module(x)
            repeated = module(x)
            torch.testing.assert_close(repeated, split, rtol=0, atol=0)
            torch.testing.assert_close(split, standard, rtol=0.01, atol=2.0)
            assert (split - standard).abs().float().mean().item() <= 0.003

        module.paroquant_triton_megakernel_decode_splitk_enabled = False
        fp16_standard_reused = module(inputs[torch.float16])
        module.paroquant_triton_megakernel_decode_splitk_enabled = True
        fp16_reused = module(inputs[torch.float16])
        fp16_repeated = module(inputs[torch.float16])

    torch.testing.assert_close(fp16_repeated, fp16_reused, rtol=0, atol=0)
    torch.testing.assert_close(fp16_reused, fp16_standard_reused, rtol=0.01, atol=2.0)
    assert (fp16_reused - fp16_standard_reused).abs().float().mean().item() <= 0.003
    assert {key[-1] for key in module._megakernel_splitk_compiled} == {torch.float16, torch.bfloat16}
    assert len(module._megakernel_splitk_scratch) == 1
    _, counters = next(iter(module._megakernel_splitk_scratch.values()))
    assert counters.numel() == out_features // 128
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K stream test")
def test_paroquant_splitk_decode_isolated_across_concurrent_streams():
    """Run real split-K work concurrently and keep scratch counters stream-local."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(seed=2601)
    x_first = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    x_second = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    module.paroquant_triton_megakernel_decode_splitk_enabled = False
    with torch.inference_mode():
        expected_first = module(x_first)
        expected_second = module(x_second)

    module.paroquant_triton_megakernel_decode_splitk_enabled = True
    module._megakernel_splitk_scratch = {}
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(first_stream):
        actual_first = module(x_first)
    with torch.inference_mode(), torch.cuda.stream(second_stream):
        actual_second = module(x_second)
    torch.cuda.current_stream().wait_stream(first_stream)
    torch.cuda.current_stream().wait_stream(second_stream)

    torch.testing.assert_close(actual_first, expected_first, rtol=0.01, atol=2.0)
    torch.testing.assert_close(actual_second, expected_second, rtol=0.01, atol=2.0)
    assert (actual_first - expected_first).abs().float().mean().item() <= 0.003
    assert (actual_second - expected_second).abs().float().mean().item() <= 0.003
    assert len(module._megakernel_splitk_scratch) == 2
    partial_ptrs = {partials.data_ptr() for partials, _ in module._megakernel_splitk_scratch.values()}
    assert len(partial_ptrs) == 2
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant live-buffer test")
def test_paroquant_splitk_decode_observes_live_buffer_updates():
    """Keep direct buffer lookup mutation-safe across warm compiled launches."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(seed=2651)
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        baseline = module(x)
    previous_rotation_key = module._megakernel_rotation_cache_key
    previous_scale_ptr = module.scales.data_ptr()

    module.scales = (module.scales.to(dtype=torch.float16) * 0.75).contiguous()
    module.bias = (module.bias.to(dtype=torch.float16) + 0.25).contiguous()
    module.channel_scales = (module.channel_scales.to(dtype=torch.float16) * 1.125).contiguous()
    with torch.no_grad():
        module.theta.add_(0.01)

    with torch.inference_mode():
        actual = module(x)
        metadata = module._megakernel_rotation_metadata(x, decode=True)
        assert metadata is not None
        partner, cos, sin, channel_scales = metadata
        expected = _paroquant_rotation_gemm_triton(
            x,
            module.qweight,
            module.scales,
            module.qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            module.bias,
            block_size_m=8,
            block_size_n=128,
            num_warps=8,
            num_stages=2,
            loop_unroll_factor=1,
            explicit_fma=True,
            prefetch_first_partner=True,
            prefetch_packed_weight=True,
        )

    cross = (actual - expected).abs().float()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=2.0)
    assert cross.mean().item() <= 0.003
    assert not torch.equal(actual, baseline)
    assert module.scales.dtype == torch.bfloat16
    assert module.bias.dtype == torch.bfloat16
    assert module.scales.data_ptr() != previous_scale_ptr
    assert module._megakernel_rotation_cache_key != previous_rotation_key
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K graph test")
def test_paroquant_splitk_decode_uses_graph_owned_scratch(monkeypatch):
    """Give each warmed capture private split-K scratch and retain the cold fallback."""
    triton = pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = _make_splitk_triton_module(seed=2701)
    static_x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    module.paroquant_triton_megakernel_decode_splitk_enabled = False
    with torch.inference_mode():
        standard = module(static_x)
    module.paroquant_triton_megakernel_decode_splitk_enabled = True
    with torch.inference_mode():
        split = module(static_x)
        repeated_split = module(static_x)
    torch.testing.assert_close(split, standard, rtol=0.01, atol=2.0)
    torch.testing.assert_close(repeated_split, split, rtol=0, atol=0)
    assert (split - standard).abs().float().mean().item() <= 0.003
    assert len(module._megakernel_splitk_scratch) == 1
    assert len(module._megakernel_splitk_compiled) == 1
    assert len(module._megakernel_splitk_prepared) == 1
    serialized_state = module.__getstate__()
    assert serialized_state["_megakernel_splitk_last_scratch"] is None
    assert serialized_state["_megakernel_splitk_compiled"] == {}
    assert serialized_state["_megakernel_splitk_prepared"] == {}

    shaped_x = static_x.unsqueeze(0)
    direct_inputs = []
    original_prepared_launch = paroquant_triton_qlinear._paroquant_rotation_gemm_splitk_triton_prepared

    def record_prepared_input(prepared_launch, input, *args, **kwargs):
        direct_inputs.append(input)
        return original_prepared_launch(prepared_launch, input, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            paroquant_triton_qlinear,
            "_paroquant_rotation_gemm_splitk_triton_prepared",
            record_prepared_input,
        )
        with torch.inference_mode():
            shaped_first = module(shaped_x)
            shaped_second = module(shaped_x)
    torch.testing.assert_close(shaped_first, split.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(shaped_second, shaped_first, rtol=0, atol=0)
    assert [input is shaped_x for input in direct_inputs] == [True, True]
    assert shaped_first.shape == (1, 1, module.out_features)
    assert shaped_first._base is None and shaped_second._base is None
    assert shaped_first.data_ptr() != shaped_second.data_ptr()
    eager_partials, eager_counters = next(iter(module._megakernel_splitk_scratch.values()))
    assert module._megakernel_splitk_last_scratch is not None
    assert module._megakernel_splitk_last_scratch[-2] is eager_partials
    assert module._megakernel_splitk_last_scratch[-1] is eager_counters

    hook_calls = {"enter": [], "exit": []}

    def enter_hook(*args):
        hook_calls["enter"].append(args)

    def exit_hook(*args):
        hook_calls["exit"].append(args)

    triton.knobs.runtime.launch_enter_hook.add(enter_hook)
    triton.knobs.runtime.launch_exit_hook.add(exit_hook)
    try:
        with torch.inference_mode():
            hooked_split = module(static_x)
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(enter_hook)
        triton.knobs.runtime.launch_exit_hook.remove(exit_hook)
    torch.testing.assert_close(hooked_split, split, rtol=0, atol=0)
    assert len(hook_calls["enter"]) == 1
    assert len(hook_calls["exit"]) == 1
    assert len(hook_calls["enter"][0]) == 1 and hook_calls["enter"][0][0] is not None

    guarded_devices = []
    original_device_context = paroquant_triton_gemm.get_same_device_cm

    def record_device_context(tensor):
        guarded_devices.append(tensor.device)
        return original_device_context(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_device", lambda: module.qweight.device.index + 1)
        patch.setattr(paroquant_triton_gemm, "get_same_device_cm", record_device_context)
        with torch.inference_mode():
            guarded_split = module(static_x)
    torch.testing.assert_close(guarded_split, split, rtol=0, atol=0)
    assert guarded_devices == [module.qweight.device]

    cache_key = module._plan_cache_key("decode", static_x)
    fallback_calls = []
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "_forward_triton_megakernel",
            lambda _kind, _x_flat, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced mega-kernel failure")),
        )

        def fallback_run(plan, _x_flat):
            fallback_calls.append(plan)
            if plan == "cuda_awq":
                return standard
            raise AssertionError(f"unexpected fallback plan: {plan}")

        patch.setattr(module, "_run_plan", fallback_run)
        with torch.inference_mode():
            fallback = module(static_x)
    torch.testing.assert_close(fallback, standard, rtol=0, atol=0)
    assert fallback_calls == ["cuda_awq"]
    assert module._plan_cache[cache_key] == "cuda_awq"
    module._plan_cache[cache_key] = "decode_megakernel"

    module.paroquant_triton_megakernel_decode_compiled_launch_enabled = False
    with torch.inference_mode():
        jit_fallback = module(static_x)
    torch.testing.assert_close(jit_fallback, split, rtol=0, atol=0)
    module.paroquant_triton_megakernel_decode_compiled_launch_enabled = True

    scratch_results = []
    original_scratch = module._megakernel_decode_splitk_scratch

    def record_scratch(x_flat, **kwargs):
        scratch = original_scratch(x_flat, **kwargs)
        scratch_results.append(scratch)
        return scratch

    monkeypatch.setattr(module, "_megakernel_decode_splitk_scratch", record_scratch)
    with torch.inference_mode():
        graphs = []
        captured_outputs = []
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = module(static_x)
            graphs.append(graph)
            captured_outputs.append(captured)
        for _ in range(3):
            graphs[0].replay()
            graphs[1].replay()
        first_graph_stream = torch.cuda.Stream()
        second_graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(first_graph_stream):
            graphs[0].replay()
        with torch.cuda.stream(second_graph_stream):
            graphs[1].replay()
        torch.cuda.current_stream().wait_stream(first_graph_stream)
        torch.cuda.current_stream().wait_stream(second_graph_stream)
        actual_outputs = [captured.clone() for captured in captured_outputs]

    assert len(scratch_results) == 2
    assert all(scratch is not None for scratch in scratch_results)
    graph_partial_ptrs = {scratch[0].data_ptr() for scratch in scratch_results}
    assert len(graph_partial_ptrs) == 2
    assert eager_partials.data_ptr() not in graph_partial_ptrs
    assert module._megakernel_splitk_last_scratch[-2] is eager_partials
    assert module._megakernel_splitk_last_scratch[-1] is eager_counters
    for actual in actual_outputs:
        torch.testing.assert_close(actual, split, rtol=0, atol=0)
    assert all(scratch[1].count_nonzero().item() == 0 for scratch in scratch_results)
    assert len(module._megakernel_splitk_scratch) == 1
    assert all(counters.count_nonzero().item() == 0 for _, counters in module._megakernel_splitk_scratch.values())

    module._megakernel_splitk_compiled = {}
    scratch_results.clear()
    with torch.inference_mode():
        cold_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold_graph):
            cold_captured = module(static_x)
        cold_graph.replay()
        cold_actual = cold_captured.clone()
    assert scratch_results == [None]
    torch.testing.assert_close(cold_actual, standard, rtol=0, atol=0)

    monkeypatch.setattr(paroquant_triton_gemm, "_paroquant_splitk_compiled_launch_supported", lambda _kernel: False)
    with torch.inference_mode():
        abi_fallback = module(static_x)
    torch.testing.assert_close(abi_fallback, split, rtol=0, atol=0)
    assert list(module._megakernel_splitk_compiled.values()) == [False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant megakernel dispatch test")
def test_paroquant_megakernel_dispatch_gate_and_grad_fallback(monkeypatch):
    """Keep the sm_80 specialization narrow and retain the autograd-safe route."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = ParoQuantTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=256,
        out_features=256,
        bias=False,
        register_buffers=True,
        krot=8,
    ).cuda().eval()
    module.paroquant_triton_autotune_enabled = False
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        assert module._select_plan("decode", x) == "decode_megakernel"
        decode_metadata = module._megakernel_rotation_metadata(x, decode=True)
        prefill_metadata = module._megakernel_rotation_metadata(x, decode=False)
        assert decode_metadata is not None and prefill_metadata is not None
        assert decode_metadata[0].dtype == torch.int16
        assert prefill_metadata[0].dtype == torch.int32
        torch.testing.assert_close(
            decode_metadata[0].to(dtype=torch.int32),
            torch.remainder(prefill_metadata[0], module.group_size),
        )
        x_bf16 = x.to(dtype=torch.bfloat16)
        bf16_decode_metadata = module._megakernel_rotation_metadata(x_bf16, decode=True)
        bf16_prefill_metadata = module._megakernel_rotation_metadata(x_bf16, decode=False)
        assert bf16_decode_metadata is not None and bf16_prefill_metadata is not None
        assert bf16_decode_metadata[0].dtype == torch.int8
        torch.testing.assert_close(
            bf16_decode_metadata[0].to(dtype=torch.int32),
            torch.remainder(bf16_prefill_metadata[0], module.group_size),
        )
        module.clear_autotune()
        module.paroquant_triton_megakernel_max_k = 128
        assert module._select_plan("decode", x) == "decode_fused"
        module.clear_autotune()
        module.paroquant_triton_megakernel_max_k = 2048
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 9))
        assert module._select_plan("decode", x) == "decode_fused"

    module.paroquant_triton_megakernel_max_k = 2048
    assert module._select_plan("decode", x) == "decode_fused"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant extended-prefill gate test")
def test_paroquant_extended_prefill_gate_requires_measured_sm_count(monkeypatch):
    """Keep N=2176-4096 prefill eligibility on the measured FP16 124-SM configuration."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = ParoQuantTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=2560,
        bias=False,
        register_buffers=True,
        krot=8,
    ).cuda().eval()
    x = torch.empty((97, 2048), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        device_properties = type(
            "DeviceProperties",
            (),
            {"multi_processor_count": 124, "major": 8, "minor": 0},
        )()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
        assert module._megakernel_ready(x)
        assert not module._megakernel_ready(x.to(dtype=torch.bfloat16))
        assert not module._megakernel_ready(torch.empty((193, 2048), device="cuda", dtype=torch.float16))

        device_properties = type(
            "DeviceProperties",
            (),
            {"multi_processor_count": 108, "major": 8, "minor": 0},
        )()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
        assert not module._megakernel_ready(x)

        device_properties = type(
            "DeviceProperties",
            (),
            {"multi_processor_count": 124, "major": 8, "minor": 0},
        )()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
        module.paroquant_triton_megakernel_prefill_max_n = 2048
        assert not module._megakernel_ready(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant large-K gate test")
def test_paroquant_large_k_splitk_gate_requires_exact_qwen_shape_and_measured_sm_count(monkeypatch):
    """Keep the Qwen-width route BF16-only, row-specific, capped, and restricted to the measured GPU."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = ParoQuantTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=4096,
        out_features=1024,
        bias=False,
        register_buffers=True,
        krot=8,
    ).cuda().eval()
    x = torch.empty((1, 4096), device="cuda", dtype=torch.bfloat16)
    device_properties = type(
        "DeviceProperties",
        (),
        {"multi_processor_count": 124, "major": 8, "minor": 0},
    )()

    with torch.inference_mode():
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
        assert module._megakernel_ready(x)
        assert module._megakernel_decode_splitk_factor(x, rows=1) == 32
        for rows in (2, 4, 8, 16, 32):
            assert module._megakernel_ready(torch.empty((rows, 4096), device="cuda", dtype=torch.bfloat16))
        for rows in (3, 64):
            assert not module._megakernel_ready(torch.empty((rows, 4096), device="cuda", dtype=torch.bfloat16))
        assert not module._megakernel_ready(x.to(dtype=torch.float16))

        module.paroquant_triton_megakernel_max_k = 2048
        assert not module._megakernel_ready(x)
        module.paroquant_triton_megakernel_max_k = 12288

        device_properties = type(
            "DeviceProperties",
            (),
            {"multi_processor_count": 108, "major": 8, "minor": 0},
        )()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
        assert not module._megakernel_ready(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant split-K scratch test")
def test_paroquant_splitk_scratch_is_stream_owned_and_narrowly_gated(monkeypatch):
    """Keep split-K scratch isolated per stream and preserve graph/non-target fallbacks."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    module = ParoQuantTritonLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=2048,
        bias=False,
        register_buffers=True,
        krot=8,
    ).cuda().eval()
    x = torch.empty((1, 2048), device="cuda", dtype=torch.bfloat16)
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)

    default_scratch = module._megakernel_decode_splitk_scratch(x)
    assert default_scratch is not None
    assert default_scratch[0].numel() == 16 * 16 * 8 * 128
    assert default_scratch[1].numel() == 16

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_scratch = module._megakernel_decode_splitk_scratch(x)
    assert stream_scratch is not None
    assert stream_scratch[0].data_ptr() != default_scratch[0].data_ptr()
    assert len(module._megakernel_splitk_scratch) == 2

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert module._megakernel_decode_splitk_scratch(x) is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    module.paroquant_triton_megakernel_decode_splitk_enabled = False
    assert module._megakernel_decode_splitk_scratch(x) is None
    module.paroquant_triton_megakernel_decode_splitk_enabled = True
    fp16_scratch = module._megakernel_decode_splitk_scratch(x.to(dtype=torch.float16))
    assert fp16_scratch is not None
    assert fp16_scratch[0] is default_scratch[0]
    assert module._megakernel_decode_splitk_scratch(torch.empty((9, 2048), device="cuda", dtype=torch.float16)) is None
    assert module._megakernel_decode_splitk_scratch(x.to(dtype=torch.float32)) is None

    module._megakernel_splitk_scratch = {}
    device_properties = type("DeviceProperties", (), {"multi_processor_count": 108})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    assert module._megakernel_decode_splitk_scratch(x) is None

    device_properties = type("DeviceProperties", (), {"multi_processor_count": 124})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device=None: device_properties)
    monkeypatch.setattr(paroquant_triton_gemm, "triton_driver", None)
    fallback_scratch = module._megakernel_decode_splitk_scratch(x)
    assert fallback_scratch is not None
    fallback_stream_id = torch.cuda.current_stream(x.device).cuda_stream
    assert any(key[1] == fallback_stream_id for key in module._megakernel_splitk_scratch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant inference-tensor test")
def test_paroquant_megakernel_accepts_inference_created_buffers():
    """Allow model construction under inference mode, whose tensors have no version counter."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    with torch.inference_mode():
        module = ParoQuantTritonLinear(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            in_features=256,
            out_features=256,
            bias=False,
            register_buffers=True,
            krot=8,
        ).cuda().eval()
        module.qweight.zero_()
        module.qzeros.zero_()
        module.scales.fill_(1)
        module.post_init()
        module.paroquant_triton_autotune_enabled = False
        x = torch.randn(1, 256, device="cuda", dtype=torch.float16)

        assert module._select_plan("decode", x) == "decode_megakernel"
        actual = module(x)

    assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant megakernel graph test")
def test_paroquant_megakernel_cuda_graph_replay():
    """Keep the warmed one-launch path safe for CUDA graph capture and replay."""
    pytest.importorskip("triton")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("ParoQuant rotation+GEMM megakernel is currently dispatched only on sm_80.")

    bits = 4
    in_features = 256
    out_features = 256
    group_size = 128
    qweight, qzeros, scales, bias = _make_packed_buffers(bits, in_features, out_features, group_size)
    module = ParoQuantTritonLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    ).cuda().eval()
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        device="cuda",
        dtype=torch.float16,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    module.qweight.copy_(qweight.cuda())
    module.qzeros.copy_(qzeros.cuda())
    module.scales.copy_(scales.cuda())
    module.bias.copy_(bias.cuda())
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.paroquant_triton_autotune_enabled = False

    static_x = torch.randn(1, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = module(static_x)
        assert all("megakernel" not in key for key in module.state_dict())
        module._forward_triton_decode = lambda _rotated: (_ for _ in ()).throw(
            AssertionError("CUDA graph capture unexpectedly left the warmed megakernel plan.")
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(static_x)
        graph.replay()
        actual = captured.clone()

    torch.testing.assert_close(actual, expected, atol=0.5, rtol=5e-3)
