# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Marlin keeps a checkpoint bias at its source width through the epilogue.

``MarlinLinear`` / ``AwqMarlinLinear`` rewrote ``bias`` twice: ``post_init``
permuted and padded it in place, and ``forward`` narrowed it to the activation
dtype. Both steps destroyed a finite checkpoint value the 16-bit dtype cannot
hold (an FP32 ``-65536`` is ``-inf`` as FP16), even though the kernel adds the
bias to its FP32 accumulator. The registered buffer is now checkpoint state;
the padded, permuted FP32 operand is derived per launch.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.utils.marlin import (
    marlin_import_exception,
    marlin_pad_dim,
    marlin_permute_bias,
    marlin_runtime_available,
    marlin_runtime_error,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or bool(torch.version.hip)
    or torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin requires NVIDIA CUDA compute capability >= 8.0",
)

DEVICE = torch.device("cuda:0")

# ``-65536`` is finite in FP32/BF16 but ``-inf`` in FP16, ``65520`` is the FP16
# overflow midpoint, and ``-1e-45`` underflows FP16 to zero.
BIAS_VALUES = (-65536.0, -65520.0, 65504.0, 65520.0, -100000.0, -1e-45, 0.0)


def _require_runtime(dtype: torch.dtype) -> None:
    if marlin_import_exception is not None:
        pytest.skip(f"Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(dtype):
        pytest.skip(marlin_runtime_error(dtype))


def _pack_gptq_columns(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack one input column per row, as the GPTQ checkpoint layout does."""
    order = (0, 2, 4, 6, 1, 3, 5, 7) if bits == 4 else (0, 1, 2, 3)
    packed = torch.zeros(
        (values.shape[0], values.shape[1] // len(order)), dtype=torch.int32
    )
    for lane, source in enumerate(order):
        packed |= values[:, source::len(order)] << (lane * bits)
    return packed


def _pack_awq(codes: torch.Tensor, zeros: torch.Tensor, bits: int):
    """Pack both AWQ tables with the shared helper (same bytes as a checkpoint)."""
    from gptqmodel.quantization.awq.utils.packing_utils import pack_awq

    return pack_awq(codes, zeros, bits)


def _layer(cls, *, k: int, n: int, group: int, dtype: torch.dtype):
    # ``register_buffers`` is only needed for the AWQ packer path; the GPTQ
    # kernel allocates its own parameters.
    return cls(
        bits=4,
        group_size=group,
        desc_act=False,
        sym=cls is MarlinLinear,
        in_features=k,
        out_features=n,
        bias=True,
        dtype=dtype,
        backend="marlin",
        register_buffers=cls is AwqMarlinLinear,
    ).to(DEVICE).eval()


def _write_unit_weight(layer, k: int, n: int, group: int) -> None:
    """Store a weight of exactly 1.0 per element.

    Symmetric GPTQ (``uint4b8``) subtracts an implicit 8 and AWQ subtracts the
    stored zero point, so code 9 with zero point 8/0 both decode to ``1 * scale``.
    ``MarlinLinear`` stores checkpoint words as ``[K / 8, N]`` and
    ``AwqMarlinLinear`` as ``[K, N / 8]``; ``post_init`` repacks both.
    """

    # Both layouts subtract the zero point before scaling, so code 9 with zero
    # point 8 decodes to ``(9 - 8) * 1.0``. GPTQ's uint4b8 uses the same
    # implicit 8 while AWQ stores it explicitly.
    codes = torch.full((k, n), 9, dtype=torch.int32)
    zeros = torch.full((k // group, n), 8, dtype=torch.int32)
    # GPTQ checkpoint words are [K / 8, N] with the 8 codes of one input column
    # in a row; AWQ words are [K, N / 8] with N packed along the row. Both use
    # the 4-bit interleave, so packing the transposed table gives the GPTQ
    # orientation directly.
    if isinstance(layer, MarlinLinear):
        packed_w = _pack_gptq_columns(codes.t().contiguous(), 4).t().contiguous()
        packed_z = _pack_gptq_columns(zeros, 4)
    else:
        packed_w, packed_z = _pack_awq(codes, zeros, 4)
    with torch.no_grad():
        assert layer.qweight.shape == packed_w.shape, (layer.qweight.shape, packed_w.shape)
        assert layer.qzeros.shape == packed_z.shape, (layer.qzeros.shape, packed_z.shape)
        layer.qweight.copy_(packed_w.to(DEVICE))
        layer.qzeros.copy_(packed_z.to(DEVICE))
        layer.scales.fill_(1.0)
        if isinstance(layer, MarlinLinear):
            layer.g_idx.copy_(
                torch.arange(k, device=DEVICE, dtype=torch.int32) // group
            )


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [MarlinLinear, AwqMarlinLinear], ids=["gptq", "awq"])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("bias_value", BIAS_VALUES, ids=[repr(v) for v in BIAS_VALUES])
def test_marlin_post_init_keeps_checkpoint_bias(cls, bias_dtype, bias_value):
    _require_runtime(torch.float16)
    k = n = 128
    layer = _layer(cls, k=k, n=n, group=128, dtype=torch.float16)
    with torch.no_grad():
        layer.register_buffer(
            "bias", torch.full((n,), bias_value, dtype=bias_dtype, device=DEVICE)
        )
    _write_unit_weight(layer, k, n, 128)
    layer.post_init()

    # The registered buffer is checkpoint state: source width, source order,
    # and no rounding or overflow. The padded/permuted form is per launch.
    assert layer.bias.dtype is bias_dtype
    assert layer.bias.shape == (n,)
    assert torch.isfinite(layer.bias).all()
    # The stored value is the checkpoint value at its own width: a BF16 source
    # rounds when it is created, and nothing here narrows it again.
    torch.testing.assert_close(
        layer.bias.cpu(),
        torch.full((n,), bias_value, dtype=bias_dtype),
        rtol=0,
        atol=0,
    )
    assert layer._marlin_bias_padding[0] == n


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [MarlinLinear, AwqMarlinLinear], ids=["gptq", "awq"])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_marlin_forward_cancels_wide_bias(cls, bias_dtype):
    """A finite bias outside the output range must cancel before the cast."""

    _require_runtime(torch.float16)
    k = n = 128
    bias_value = -65536.0
    if bias_dtype is torch.bfloat16:
        bias_value = float(torch.tensor(bias_value, dtype=torch.bfloat16))
    layer = _layer(cls, k=k, n=n, group=128, dtype=torch.float16)
    with torch.no_grad():
        layer.register_buffer(
            "bias", torch.full((n,), bias_value, dtype=bias_dtype, device=DEVICE)
        )
    _write_unit_weight(layer, k, n, 128)
    layer.post_init()
    stored = layer.bias.clone()

    x = torch.full((1, k), 512.0, dtype=torch.float16, device=DEVICE)
    with torch.inference_mode():
        actual = layer(x)
        repeated = layer(x)

    expected = 512.0 * k + bias_value
    assert actual.dtype is torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.cpu().double(),
        torch.full((1, n), expected, dtype=torch.float64),
        rtol=0,
        atol=1e-2,
    )
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    assert layer.bias.dtype is bias_dtype
    assert torch.equal(layer.bias, stored)


@pytest.mark.cuda
@pytest.mark.parametrize("cls", [MarlinLinear, AwqMarlinLinear], ids=["gptq", "awq"])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_marlin_bias_columns_follow_marlin_order(cls, bias_dtype):
    """The derived launch operand must address the right output column."""

    _require_runtime(torch.float16)
    n = 64
    layer = _layer(cls, k=128, n=n, group=128, dtype=torch.float16)
    bias = torch.arange(n, device=DEVICE, dtype=torch.float32) - 32.0
    with torch.no_grad():
        layer.register_buffer("bias", bias.to(bias_dtype))
    _write_unit_weight(layer, 128, n, 128)
    layer.post_init()

    kernel_bias = layer._kernel_bias(
        torch.empty(1, 128, device=DEVICE, dtype=torch.float16)
    )
    assert kernel_bias.dtype is torch.float32
    assert torch.equal(kernel_bias, marlin_permute_bias(marlin_pad_dim(bias, n, n)))


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("k", "n", "group"),
    [(288, 200, 32), (256, 200, 32), (208, 256, -1)],
    ids=["both-tail", "n-tail", "k-tail-channelwise"],
)
def test_marlin_padded_bias_operand_is_the_tile_permutation(k, n, group):
    """Tile padding must appear only in the launch operand, never in storage."""

    _require_runtime(torch.float16)
    layer = _layer(AwqMarlinLinear, k=k, n=n, group=group, dtype=torch.float16)
    bias = torch.arange(n, device=DEVICE, dtype=torch.float32) - 100.0
    with torch.no_grad():
        layer.register_buffer("bias", bias.clone())
    _write_unit_weight(layer, k, n, group if group > 0 else k)
    layer.post_init()

    padded_n = layer._marlin_bias_padding[1]
    assert padded_n >= n
    assert layer.bias.shape == (n,)
    kernel_bias = layer._kernel_bias(
        torch.empty(1, k, device=DEVICE, dtype=torch.float16)
    )
    assert kernel_bias.numel() == padded_n
    assert torch.equal(kernel_bias, marlin_permute_bias(marlin_pad_dim(bias, n, padded_n)))


@pytest.mark.cuda
def test_marlin_bias_derivation_is_call_local():
    """The widened/permuted operand is not retained between calls."""

    _require_runtime(torch.float16)
    layer = _layer(AwqMarlinLinear, k=128, n=128, group=128, dtype=torch.float16)
    with torch.no_grad():
        layer.register_buffer(
            "bias", torch.full((128,), -65536.0, dtype=torch.float32, device=DEVICE)
        )
    _write_unit_weight(layer, 128, 128, 128)
    layer.post_init()
    x = torch.ones(1, 128, device=DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        first = layer(x)
        second = layer(x)
    assert torch.equal(first, second)
    retained = list(vars(layer).values()) + list(layer.buffers()) + list(layer.parameters())
    assert not any(
        isinstance(value, torch.Tensor)
        and value.dtype is torch.float32
        and value is not layer.bias
        and value.shape == layer.bias.shape
        for value in retained
    )
