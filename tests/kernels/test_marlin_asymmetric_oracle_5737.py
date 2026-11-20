# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Asymmetric GPTQ Marlin against an independent float64 oracle (#5737).

``MarlinLinear`` now packs the checkpoint's logical zero points into Marlin's
``kU4``/``kU8`` runtime layout for ``sym=False``. The expected value here never
comes from the kernel or its intermediates: the packed codes, zero points and
scales are dequantized with float64 Torch arithmetic and multiplied with the
same activations. The visible drift is therefore the kernel's rounding only.

The packed tables use the checkpoint conventions: GPTQ weight words are
``[K / pack_factor, N]`` with lane ``j`` at bit ``j * bits`` and GPTQ zero words
are ``[G, N / pack_factor]`` in the same natural order.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.marlin import (
    marlin_import_exception,
    marlin_runtime_available,
    marlin_runtime_error,
)


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the Marlin asymmetric numerical test"
    if torch.version.hip:
        return "Marlin is CUDA-only"
    if torch.cuda.get_device_capability() < (8, 0):
        return "Marlin requires NVIDIA CUDA compute capability >= 8.0"
    if marlin_import_exception is not None:
        return f"Marlin kernel unavailable: {marlin_import_exception}"
    return None


pytestmark = pytest.mark.skipif(
    _skip_reason() is not None, reason=_skip_reason() or ""
)

DEVICE = torch.device("cuda:0")

# 2e-3 relative plus four output-dtype ULPs of absolute slack. The oracle is
# float64, so this separates kernel arithmetic from the mandatory rounding of
# the visible FP16/BF16 result.
RTOL = 2e-3


def _pack_k(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[K, N]`` codes into GPTQ ``[K / pack_factor, N]`` weight words."""

    k, n = codes.shape
    pack_factor = 32 // bits
    shifts = (torch.arange(pack_factor, dtype=torch.int64) * bits).view(pack_factor, 1)
    words = (codes.reshape(k // pack_factor, pack_factor, n).to(torch.int64) << shifts).sum(1)
    return words.to(torch.int32)


def _pack_n(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``[G, N]`` codes into GPTQ ``[G, N / pack_factor]`` zero words."""

    groups, n = codes.shape
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int64) * bits
    words = (
        codes.reshape(groups, n // pack_factor, pack_factor).to(torch.int64) << shifts
    ).sum(-1)
    return words.to(torch.int32)


def _oracle(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize and multiply in float64, independent of every kernel."""

    repeat = group_size if group_size > 0 else codes.shape[0]
    logical_zeros = zeros.repeat_interleave(repeat, dim=0).double()
    logical_scales = scales.repeat_interleave(repeat, dim=0).double()
    weight = (codes.double() - logical_zeros) * logical_scales
    return x.double().cpu() @ weight


def _make_case(bits: int, group_size: int, dtype: torch.dtype, *, k: int = 256,
               n: int = 128, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    maxq = (1 << bits) - 1
    groups = k // group_size if group_size > 0 else 1
    codes = torch.randint(0, maxq + 1, (k, n), generator=generator, dtype=torch.int32)
    zeros = torch.randint(0, maxq + 1, (groups, n), generator=generator, dtype=torch.int32)
    scales = ((torch.rand((groups, n), generator=generator) * 0.004) + 1e-4).to(dtype)
    x = torch.randn(4, k, generator=generator).to(dtype)

    layer = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=k,
        out_features=n,
        bias=False,
        dtype=dtype,
        backend=BACKEND.MARLIN,
    ).to(DEVICE, dtype)
    with torch.no_grad():
        layer.qweight.copy_(_pack_k(codes, bits).to(DEVICE))
        layer.qzeros.copy_(_pack_n(zeros, bits).to(DEVICE))
        layer.scales.copy_(scales.to(DEVICE))
        layer.g_idx.copy_(
            torch.arange(k, device=DEVICE, dtype=torch.int32)
            // (group_size if group_size > 0 else k)
        )
        layer.qzero_format(format=2)
        layer.post_init()
        actual = layer(x.to(DEVICE))
    return layer, codes, zeros, scales, x, actual


def _assert_matches_oracle(actual, expected, dtype, label):
    drift = (actual.double().cpu() - expected).abs().max().item()
    atol = 4 * torch.finfo(dtype).eps * max(1.0, expected.abs().max().item())
    print(
        f"{label}: max_abs_drift={drift:.3e} atol={atol:.3e} "
        f"max_abs_output={expected.abs().max().item():.3f}"
    )
    torch.testing.assert_close(
        actual.double().cpu(),
        expected,
        rtol=RTOL,
        atol=atol,
        msg=lambda message: f"{label} ({message})",
    )
    return drift


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_asymmetric_marlin_matches_float64_oracle(bits, group_size, dtype):
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    if not marlin_runtime_available(dtype):
        pytest.skip(marlin_runtime_error(dtype))

    layer, codes, zeros, scales, x, actual = _make_case(bits, group_size, dtype)
    expected = _oracle(codes, zeros, scales, x, group_size)

    assert actual.dtype is dtype
    _assert_matches_oracle(actual, expected, dtype, f"bits={bits} g={group_size} {dtype}")

    # The kernel's own result must be repeatable from the packed tensors.
    with torch.inference_mode():
        repeated = layer(x.to(DEVICE))
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.cuda
def test_asymmetric_marlin_matches_the_portable_torch_kernel():
    """The same checkpoint words must decode identically on the portable path."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    k, n, group, bits, dtype = 256, 128, 128, 4, torch.float16
    layer, codes, zeros, scales, x, actual = _make_case(bits, group, dtype)
    expected = _oracle(codes, zeros, scales, x, group)

    reference = TorchLinear(
        bits=bits,
        group_size=group,
        desc_act=False,
        sym=False,
        in_features=k,
        out_features=n,
        bias=False,
        dtype=dtype,
        backend=BACKEND.GPTQ_TORCH,
    ).to(DEVICE, dtype)
    with torch.no_grad():
        reference.qweight.copy_(_pack_k(codes, bits).to(DEVICE))
        reference.qzeros.copy_(_pack_n(zeros, bits).to(DEVICE))
        reference.scales.copy_(scales.to(DEVICE))
        reference.g_idx.copy_(torch.arange(k, device=DEVICE, dtype=torch.int32) // group)
        reference.qzero_format(format=2)
        reference.post_init()
        portable = reference(x.to(DEVICE))

    _assert_matches_oracle(portable, expected, dtype, "portable torch")
    torch.testing.assert_close(portable, actual, rtol=RTOL, atol=4 * torch.finfo(dtype).eps)


@pytest.mark.cuda
def test_asymmetric_marlin_converts_v1_zero_points_before_packing():
    """A v1 checkpoint (stored zero minus one) reaches the same result."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

    k, n, group, bits, dtype = 256, 128, 128, 4, torch.float16
    generator = torch.Generator().manual_seed(11)
    maxq = (1 << bits) - 1
    groups = k // group
    codes = torch.randint(0, maxq + 1, (k, n), generator=generator, dtype=torch.int32)
    # v1 stores ``zero - 1``; keep zeros >= 1 so the stored field is valid.
    zeros = torch.randint(1, maxq + 1, (groups, n), generator=generator, dtype=torch.int32)
    scales = ((torch.rand((groups, n), generator=generator) * 0.004) + 1e-4).to(dtype)
    x = torch.randn(4, k, generator=generator).to(dtype)

    layer = MarlinLinear(
        bits=bits,
        group_size=group,
        desc_act=False,
        sym=False,
        in_features=k,
        out_features=n,
        bias=False,
        dtype=dtype,
        backend=BACKEND.MARLIN,
    ).to(DEVICE, dtype)
    with torch.no_grad():
        layer.qweight.copy_(_pack_k(codes, bits).to(DEVICE))
        layer.qzeros.copy_(_pack_n(zeros - 1, bits).to(DEVICE))
        layer.scales.copy_(scales.to(DEVICE))
        assert layer.qzero_format() == 1
        convert_gptq_v1_to_v2_format_module(layer, bits, torch.int32)
        assert layer.qzero_format() == 2
        layer.post_init()
        actual = layer(x.to(DEVICE))

    expected = _oracle(codes, zeros, scales, x, group)
    _assert_matches_oracle(actual, expected, dtype, "v1 checkpoint")


@pytest.mark.cuda
def test_asymmetric_marlin_matches_oracle_for_tile_padded_shapes():
    """A tile-misaligned N is zero padded; the logical columns stay exact."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    layer, codes, zeros, scales, x, actual = _make_case(
        4, 128, torch.float16, n=96
    )
    expected = _oracle(codes, zeros, scales, x, 128)

    assert actual.shape == (4, 96)
    assert layer._marlin_tile_padding is not None
    _assert_matches_oracle(actual, expected, torch.float16, "padded n=96")


@pytest.mark.cuda
def test_asymmetric_marlin_keeps_packed_storage():
    """The zero-point path must not retain a dense or expanded weight copy."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    k, n, group, bits = 256, 128, 128, 4
    layer, *_ = _make_case(bits, group, torch.float16)

    assert layer.qweight.dtype is torch.int32
    assert layer.qzeros.dtype is torch.int32
    # Layout-agnostic packed-size check: the runtime tile layout still stores
    # exactly ``bits`` per weight and at most one byte per logical zero point.
    assert layer.qweight.numel() * layer.qweight.element_size() == k * n * bits // 8
    assert (
        layer.qzeros.numel() * layer.qzeros.element_size() <= (k // group) * n
    )
    assert layer._marlin_repacked is True
    assert layer._is_runtime_layout() is True
    floating = [
        tensor
        for tensor in list(layer.parameters()) + list(layer.buffers())
        if tensor.is_floating_point()
    ]
    assert all(tensor.numel() != k * n for tensor in floating)
