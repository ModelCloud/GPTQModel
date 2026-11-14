# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear, _undo_awq_interleave
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear, marlin_import_exception
from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.machete import _validate_machete_device_support, machete_runtime_error
from gptqmodel.utils.marlin import (
    awq_to_marlin_zero_points,
    marlin_runtime_available,
    marlin_runtime_error,
)
from gptqmodel.utils.swordfish import (
    _validate_swordfish_device_support,
    prewarm_swordfish_extension,
    swordfish_runtime_error,
)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    # Only the classic 4-bit layout is interleaved. Wider widths are packed as
    # a contiguous little-endian stream, matching AWQ readers such as
    # AwqTorch/AllSpark.
    order_map = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else list(range(pack_factor))

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (lane * bits)
    return packed


def _marlin_zero_points_reference(
    zero_points: torch.Tensor,
    size_k: int,
    size_n: int,
    bits: int,
) -> torch.Tensor:
    """Independent Torch oracle for Marlin's zero-point permutation + packing."""
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 2, 1, 3]

    zp = zero_points.reshape(-1, len(scale_perm))[:, scale_perm]
    zp = zp.reshape(-1, len(interleave))[:, interleave].ravel()
    zp = zp.reshape(size_k, size_n).to(torch.int32)

    pack_factor = 32 // bits
    fields = zp.reshape(size_k, size_n // pack_factor, pack_factor)
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    return (fields << shifts).sum(dim=2, dtype=torch.int32)


def _mock_awq_module_tensors(
    *,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = in_features // group_size
    maxq = (1 << bits) - 1

    float_weight = torch.randn(in_features, out_features, dtype=torch.float32) * 0.2
    weight_groups = float_weight.view(groups, group_size, out_features)

    w_min = weight_groups.amin(dim=1)
    w_max = weight_groups.amax(dim=1)
    scales = ((w_max - w_min).clamp_min(1e-6) / maxq).to(torch.float16)
    zero_points = torch.round((-w_min / scales.to(torch.float32))).clamp_(0, maxq).to(torch.int32)
    quantized = torch.round(
        weight_groups / scales.to(torch.float32).unsqueeze(1) + zero_points.unsqueeze(1)
    ).clamp_(0, maxq).to(torch.int32)

    qweight = _pack_awq_tensor(quantized.view(in_features, out_features), bits)
    qzeros = _pack_awq_tensor(zero_points, bits)
    bias = torch.randn(out_features, dtype=torch.float16)

    return qweight, qzeros, scales, bias


def _build_awq_module(
    module_cls,
    *,
    device: torch.device,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    dtype: torch.dtype = torch.float16,
):
    module = module_cls(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        dtype=dtype,
    ).to(device)

    with torch.no_grad():
        module.qweight.copy_(qweight.to(device))
        module.qzeros.copy_(qzeros.to(device))
        module.scales.copy_(scales.to(dtype=dtype, device=device))
        module.bias.copy_(bias.to(dtype=dtype, device=device))

    module.post_init()
    module.eval()
    return module


def _assert_awq_candidate_matches_torch(
    module_cls,
    *,
    device: torch.device,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    atol: float,
    rtol: float,
) -> None:
    torch.manual_seed(11)
    qweight, qzeros, scales, bias = _mock_awq_module_tensors(
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
    )

    baseline = _build_awq_module(
        AwqTorchLinear,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=bias,
    )
    candidate = _build_awq_module(
        module_cls,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=bias,
    )

    x = torch.randn((32, in_features), device=device, dtype=torch.float16)
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
    torch.cuda.synchronize(device)

    assert candidate.qzeros.numel() > 0
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(repeat, expected, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_marlin_cuda_zero_points_match_torch_awq():
    if marlin_import_exception is not None:
        pytest.skip(f"AWQ Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(torch.float16):
        pytest.skip(marlin_runtime_error(torch.float16))

    _assert_awq_candidate_matches_torch(
        AwqMarlinLinear,
        device=torch.device("cuda:0"),
        bits=4,
        group_size=64,
        in_features=256,
        out_features=128,
        atol=8e-3,
        rtol=8e-3,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,in_features,out_features,group_size",
    [
        (1, 256, 200, 32),  # N tail during decode
        (17, 208, 256, -1),  # K tail with channelwise scales
        (8, 208, 256, 208),  # Explicit K-sized channelwise group
        (33, 288, 200, 32),  # N and K tails
    ],
)
def test_awq_marlin_cuda_padded_shape_matches_dense_reference(
    dtype, rows, in_features, out_features, group_size
):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8 and dtype == torch.bfloat16:
        pytest.skip("AWQ Marlin BF16 requires compute capability >= 8.0")
    if marlin_import_exception is not None:
        pytest.skip(f"AWQ Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(dtype):
        pytest.skip(marlin_runtime_error(dtype))

    torch.manual_seed(23)
    device = torch.device("cuda:0")
    bits = 4
    effective_group_size = in_features if group_size == -1 else group_size
    groups = in_features // effective_group_size
    maxq = (1 << bits) - 1

    codes = torch.randint(
        0,
        maxq + 1,
        (in_features, out_features),
        dtype=torch.int32,
    )
    zero_points = torch.randint(
        0,
        maxq + 1,
        (groups, out_features),
        dtype=torch.int32,
    )
    scales = (torch.rand((groups, out_features)) * 0.01 + 0.001).to(dtype)
    bias = (torch.randn(out_features) * 0.01).to(dtype)
    qweight = _pack_awq_tensor(codes, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = _build_awq_module(
        AwqMarlinLinear,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=bias,
        dtype=dtype,
    )

    group_indices = torch.arange(in_features) // effective_group_size
    dense_weight = (
        codes - zero_points.index_select(0, group_indices)
    ).to(dtype) * scales.index_select(0, group_indices)
    dense_weight = dense_weight.to(device)
    x = torch.randn((rows, in_features), device=device, dtype=dtype) / in_features**0.5
    expected = x @ dense_weight + bias.to(device)
    with torch.inference_mode():
        actual = module(x)
        repeated = module(x)
    torch.cuda.synchronize(device)

    assert module._marlin_tile_padding is not None
    assert actual.shape == (rows, out_features)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(repeated, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_machete_cuda_zero_points_match_torch_awq():
    if not _validate_machete_device_support():
        pytest.skip(machete_runtime_error())

    _assert_awq_candidate_matches_torch(
        AwqMacheteLinear,
        device=torch.device("cuda:0"),
        bits=4,
        group_size=64,
        in_features=128,
        out_features=128,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_swordfish_cuda_zero_points_match_torch_awq():
    if not _validate_swordfish_device_support():
        pytest.skip(swordfish_runtime_error())
    prewarm_swordfish_extension()

    _assert_awq_candidate_matches_torch(
        AwqSwordfishLinear,
        device=torch.device("cuda:0"),
        bits=4,
        group_size=128,
        in_features=256,
        out_features=128,
        atol=0.15,
        rtol=0.05,
    )


def test_awq_marlin_zero_points_8bit_keeps_natural_column_order():
    """8-bit AWQ zero points are packed in natural order, so Marlin must only
    apply its own permutation and not undo a 4-bit-style interleave."""
    torch.manual_seed(3974)
    bits = 8
    size_k, size_n = 4, 256
    zero_points = torch.randint(0, 1 << bits, (size_k, size_n), dtype=torch.int32)

    packed = _pack_awq_tensor(zero_points, bits)
    actual = awq_to_marlin_zero_points(packed, size_k, size_n, bits)
    expected = _marlin_zero_points_reference(zero_points, size_k, size_n, bits)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_machete_undo_awq_interleave_8bit_is_identity():
    values = torch.arange(64, dtype=torch.int32).reshape(8, 8)

    actual = _undo_awq_interleave(values, 8)

    torch.testing.assert_close(actual, values, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", [128, -1])
def test_awq_marlin_cuda_8bit_matches_dense_reference(group_size):
    if marlin_import_exception is not None:
        pytest.skip(f"AWQ Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(torch.float16):
        pytest.skip(marlin_runtime_error(torch.float16))

    torch.manual_seed(3974)
    device = torch.device("cuda:0")
    dtype = torch.float16
    bits = 8
    rows, in_features, out_features = 8, 512, 256
    effective_group_size = in_features if group_size == -1 else group_size
    groups = in_features // effective_group_size
    maxq = (1 << bits) - 1

    codes = torch.randint(0, maxq + 1, (in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, maxq + 1, (groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features) * 0.01 + 0.001).to(dtype)
    qweight = _pack_awq_tensor(codes, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)

    module = _build_awq_module(
        AwqMarlinLinear,
        device=device,
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bias=torch.zeros(out_features, dtype=dtype),
    )

    group_indices = torch.arange(in_features) // effective_group_size
    dense_weight = (codes - zero_points.index_select(0, group_indices)).to(dtype)
    dense_weight = dense_weight * scales.index_select(0, group_indices)
    dense_weight = dense_weight.to(device)
    x = torch.randn((rows, in_features), device=device, dtype=dtype) / in_features**0.5
    expected = x @ dense_weight

    with torch.inference_mode():
        actual = module(x)
        repeated = module(x)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(repeated, expected, atol=5e-2, rtol=5e-2)
