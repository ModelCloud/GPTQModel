# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fused-decode W*A16 GEMMs for fine-grained checkpoints (NVFP4, MXFP8).

Inkling 2 Mini stores its routed experts as NVFP4 (layers 0-19) and MXFP8
(20-23) 3-D parameters. Dequantizing them to BF16 costs ~49 GB for the expert
stack alone, so both formats get a Triton GEMM that decodes the weight tile
inside the K loop and never materializes the dense matrix. Neither kernel needs
`tl.dot_scaled`, CUTLASS or Marlin's scale permutation, so they run on every
Ampere-and-newer part with a plain BF16/FP16 `tl.dot` and FP32 accumulation.

The NVFP4 contract (NVIDIA ModelOpt, `quant_algo: NVFP4`):

    value = e2m1(code) * e4m3(block_scale) * global_scale

with two codes per byte (the even column in the low nibble), one E4M3 scale per
16 contiguous columns and one FP32 scale per tensor.

Every decode test runs against varied E4M3 block scales. A uniform scale makes a
wrong-group read indistinguishable from a right one, which is exactly how the
Marlin group-16 layout bug stayed hidden.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils.nvfp4_triton import (
    MXFP8_GROUP,
    NVFP4_GROUP,
    TRITON_AVAILABLE,
    dequant_mxfp8,
    dequant_nvfp4,
    mxfp8_w8a16_linear,
    nvfp4_w4a16_linear,
    scaled_wxa16_available,
)


_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
# Real expert shapes from the checkpoint: w13 is [2304, 2304], w2 [2304, 1152].
_SHAPES = [(2304, 2304), (2304, 1152), (576, 1152), (256, 512), (100, 96)]
_M = [1, 7, 16, 33, 128, 144]


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the fused-decode Triton GEMMs"
    if not TRITON_AVAILABLE:
        return "Triton is not installed"
    if not scaled_wxa16_available(torch.device("cuda", torch.cuda.current_device())):
        return "fused-decode GEMM unavailable on this device"
    return None


_SKIP = _skip_reason()


def _nvfp4_operands(size_n: int, size_k: int, device: torch.device, seed: int):
    """Packed NVFP4 weight with a distinct E4M3 scale per 16-wide group."""

    g = torch.Generator().manual_seed(seed)
    nibbles = torch.randint(0, 16, (size_n, size_k), generator=g)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)  # low nibble = even k
    # 0.5 .. ~448, every value a legal finite E4M3 code.
    raw = torch.randint(0, 127, (size_n, size_k // NVFP4_GROUP), generator=g, dtype=torch.int32)
    scales = raw.to(torch.uint8).view(torch.float8_e4m3fn)
    return packed.to(device), scales.to(device), _E2M1[nibbles]


def _nvfp4_reference(size_n: int, size_k: int, device: torch.device, seed: int):
    packed, scales, values = _nvfp4_operands(size_n, size_k, device, seed)
    dense = (values * scales.double().cpu().repeat_interleave(NVFP4_GROUP, dim=1)).to(device)
    return packed, scales, dense


def _gemm(x, packed, scales, global_scale, out_dtype=None):
    return nvfp4_w4a16_linear(x, packed, scales, global_scale, out_dtype=out_dtype)


def _fro(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


# --------------------------------------------------------------------------- #
# NVFP4 decode
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
@pytest.mark.parametrize("size_n,size_k", [(128, 256), (64, 1024)])
def test_nvfp4_weight_decode_is_bit_exact(size_n, size_k):
    """The identity probe recovers the decoded BF16 weight with no error at all."""

    device = torch.device("cuda", torch.cuda.current_device())
    packed, scales, dense = _nvfp4_reference(size_n, size_k, device, seed=size_k)
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)
    recovered = _gemm(identity, packed, scales, 1.0).T.contiguous()
    # `dense` is FP64; the kernel rounds each weight to BF16 exactly like the
    # dense path does before a BF16 matmul.
    assert torch.equal(recovered, dense.to(torch.bfloat16))


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_low_nibble_is_the_even_column():
    """One-hot codes: the low nibble is column 2i, the high nibble column 2i+1."""

    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k = 4, 64
    scales = torch.ones(size_n, size_k // NVFP4_GROUP, device=device).to(torch.float8_e4m3fn)
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)

    low = torch.zeros(size_n, size_k // 2, dtype=torch.uint8, device=device)
    low[:, 0] = 0x2  # magnitude 1.0 in the low nibble of byte 0
    row = _gemm(identity, low, scales, 1.0)[:, 0]
    assert row.nonzero().flatten().tolist() == [0]
    assert row[0].item() == pytest.approx(1.0)

    high = torch.zeros(size_n, size_k // 2, dtype=torch.uint8, device=device)
    high[:, 0] = 0x20
    row = _gemm(identity, high, scales, 1.0)[:, 0]
    assert row.nonzero().flatten().tolist() == [1]
    assert row[1].item() == pytest.approx(1.0)

    # A negative code keeps its sign in the high nibble.
    signed = torch.zeros(size_n, size_k // 2, dtype=torch.uint8, device=device)
    signed[:, 3] = 0xA0  # -1.0 in the high nibble of byte 3 -> column 7
    row = _gemm(identity, signed, scales, 1.0)[:, 0]
    assert row.nonzero().flatten().tolist() == [7]
    assert row[7].item() == pytest.approx(-1.0)


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_block_scale_covers_exactly_16_columns():
    """A scale must apply to its own 16 columns and to no neighbour's."""

    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k = 2, 128
    scales = torch.ones(size_n, size_k // NVFP4_GROUP, device=device)
    scales[:, 1] = 3.0
    scales[:, 3] = 0.25
    scales = scales.to(torch.float8_e4m3fn)
    packed = torch.full((size_n, size_k // 2), 0x22, dtype=torch.uint8, device=device)  # 1.0 everywhere
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)
    recovered = _gemm(identity, packed, scales, 1.0)[:, 0]
    expected = torch.ones(size_k, device=device)
    expected[16:32] = 3.0
    expected[48:64] = 0.25
    assert torch.equal(recovered, expected)


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_every_finite_e4m3_scale_byte_decodes():
    """All 254 finite E4M3 scale bytes through the kernel's own decoder.

    0x7F/0xFF are the format's NaNs, and a NaN weight anywhere in a row makes
    every dot product that reads it NaN, so they get their own test.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    raw = torch.tensor([b for b in range(256) if (b & 0x7F) != 0x7F], dtype=torch.uint8)
    size_k = NVFP4_GROUP * raw.numel()
    scales = raw.view(torch.float8_e4m3fn).reshape(1, -1).to(device)
    packed = torch.full((1, size_k // 2), 0x22, dtype=torch.uint8, device=device)  # 1.0 everywhere
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)
    recovered = _gemm(identity, packed, scales, 1.0)[:, 0][::NVFP4_GROUP]
    assert torch.equal(recovered.double(), scales[0].double())


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_nan_scale_byte_propagates():
    """0x7F/0xFF are NaN scales, not the largest finite E4M3 values."""

    device = torch.device("cuda", torch.cuda.current_device())
    for nan_byte in (0x7F, 0xFF):
        scales = torch.tensor([0x38, nan_byte], dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(1, 2)
        packed = torch.full((1, 16), 0x22, dtype=torch.uint8, device=device)
        identity = torch.eye(32, device=device, dtype=torch.bfloat16)
        assert _gemm(identity, packed, scales.to(device), 1.0).isnan().all()


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_global_scale_scales_the_output_once():
    """The FP32 global scale multiplies the accumulator, not each weight."""

    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k, rows = 64, 512, 16
    packed, scales, _ = _nvfp4_operands(size_n, size_k, device, seed=91)
    x = torch.randn(rows, size_k, device=device, dtype=torch.bfloat16)
    unit = _gemm(x, packed, scales, 1.0)
    for global_scale in (0.5, 8.0, 1 / 2688.0):
        scaled = _gemm(x, packed, scales, global_scale)
        # The scale is applied to the FP32 accumulator and rounded once, so the
        # result may differ from scaling the already-rounded output by one ulp.
        assert _fro(scaled, unit.double() * global_scale) < 5e-3
    assert torch.equal(_gemm(x, packed, scales, 0.0), torch.zeros_like(unit))


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
@pytest.mark.parametrize("size_n,size_k", _SHAPES)
@pytest.mark.parametrize("rows", _M)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_matches_fp64_oracle(size_n, size_k, rows, dtype):
    """Fused decode against a dense FP64 matmul, at BF16/FP16 GEMM resolution."""

    device = torch.device("cuda", torch.cuda.current_device())
    packed, scales, dense = _nvfp4_reference(size_n, size_k, device, seed=rows * size_k + size_n)
    x = torch.randn(rows, size_k, device=device, dtype=dtype)
    global_scale = 0.0037
    actual = _gemm(x, packed, scales, global_scale)
    expected = x.double() @ (dense * global_scale).T
    tolerance = 5e-3 if dtype == torch.bfloat16 else 5e-4
    assert actual.dtype == dtype
    assert _fro(actual, expected) < tolerance


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_uniform_scales_still_work():
    """Kept as the degenerate case; the varied-scale tests are the real check."""

    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k, rows = 256, 1024, 8
    g = torch.Generator().manual_seed(4)
    nibbles = torch.randint(0, 16, (size_n, size_k), generator=g)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8).to(device)
    scales = torch.full((size_n, size_k // NVFP4_GROUP), 1.0, device=device).to(torch.float8_e4m3fn)
    x = torch.randn(rows, size_k, device=device, dtype=torch.bfloat16)
    actual = _gemm(x, packed, scales, 1.0)
    expected = x.double() @ _E2M1[nibbles].to(device).T
    assert _fro(actual, expected) < 5e-3


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_bias_is_added_in_fp32():
    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k, rows = 96, 256, 5
    packed, scales, dense = _nvfp4_reference(size_n, size_k, device, seed=17)
    x = torch.randn(rows, size_k, device=device, dtype=torch.bfloat16)
    bias = torch.randn(size_n, device=device, dtype=torch.bfloat16)
    actual = nvfp4_w4a16_linear(x, packed, scales, 1.0, bias=bias)
    expected = x.double() @ dense.T + bias.double()
    assert _fro(actual, expected) < 5e-3


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_accepts_leading_dimensions():
    device = torch.device("cuda", torch.cuda.current_device())
    size_n, size_k = 64, 128
    packed, scales, dense = _nvfp4_reference(size_n, size_k, device, seed=23)
    x = torch.randn(2, 3, size_k, device=device, dtype=torch.bfloat16)
    actual = _gemm(x, packed, scales, 1.0)
    expected = x.double().reshape(-1, size_k) @ dense.T
    assert actual.shape == (2, 3, size_n)
    assert _fro(actual.reshape(-1, size_n), expected) < 5e-3


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_rejects_a_mismatched_scale_grid():
    device = torch.device("cuda", torch.cuda.current_device())
    packed, scales, _ = _nvfp4_operands(32, 128, device, seed=1)
    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one E4M3 block scale per 16 columns"):
        nvfp4_w4a16_linear(x, packed, scales[:, :4], 1.0)
    with pytest.raises(ValueError, match="does not match K"):
        nvfp4_w4a16_linear(x[:, :64], packed, scales, 1.0)
    with pytest.raises(ValueError, match="must be a scalar"):
        nvfp4_w4a16_linear(x, packed, scales, torch.ones(2, device=device))


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_nvfp4_torch_reference_matches_the_kernel_decode():
    device = torch.device("cuda", torch.cuda.current_device())
    packed, scales, dense = _nvfp4_reference(48, 256, device, seed=31)
    assert torch.equal(dequant_nvfp4(packed, scales, 1.0).double(), dense)


# --------------------------------------------------------------------------- #
# MXFP8 decode
# --------------------------------------------------------------------------- #


def _mxfp8_operands(size_n: int, size_k: int, device: torch.device, seed: int):
    g = torch.Generator().manual_seed(seed)
    # Every finite E4M3 code; 0x7F/0xFF are the format's NaNs.
    codes = torch.randint(0, 256, (size_n, size_k), generator=g, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes).to(torch.uint8)
    weight = codes.view(torch.float8_e4m3fn).to(device)
    exponents = torch.randint(119, 135, (size_n, size_k // MXFP8_GROUP), generator=g, dtype=torch.int32)
    scales = exponents.to(torch.uint8).to(device)
    dense = weight.double() * torch.pow(2.0, exponents.double() - 127).repeat_interleave(MXFP8_GROUP, dim=1).to(device)
    return weight, scales, dense


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
@pytest.mark.parametrize("size_n,size_k", _SHAPES)
@pytest.mark.parametrize("rows", [1, 16, 144])
def test_mxfp8_weight_decode_is_bit_exact(size_n, size_k, rows):
    device = torch.device("cuda", torch.cuda.current_device())
    weight, scales, dense = _mxfp8_operands(size_n, size_k, device, seed=size_k + rows)
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)
    recovered = mxfp8_w8a16_linear(identity, weight, scales).T.contiguous()
    assert torch.equal(recovered, dense.to(torch.bfloat16))

    x = torch.randn(rows, size_k, device=device, dtype=torch.bfloat16)
    actual = mxfp8_w8a16_linear(x, weight, scales)
    expected = x.double() @ dense.T
    assert _fro(actual, expected) < 5e-3


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_mxfp8_scale_covers_exactly_32_columns():
    device = torch.device("cuda", torch.cuda.current_device())
    size_k = 128
    weight = torch.full((1, size_k), 1.0, device=device).to(torch.float8_e4m3fn)
    scales = torch.full((1, size_k // MXFP8_GROUP), 127, dtype=torch.uint8, device=device)
    scales[0, 1] = 130  # 2 ** 3
    identity = torch.eye(size_k, device=device, dtype=torch.bfloat16)
    recovered = mxfp8_w8a16_linear(identity, weight, scales)[:, 0]
    expected = torch.ones(size_k, device=device)
    expected[32:64] = 8.0
    assert torch.equal(recovered, expected)


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_mxfp8_e8m0_nan_scale_propagates():
    device = torch.device("cuda", torch.cuda.current_device())
    weight = torch.ones((1, 32), device=device).to(torch.float8_e4m3fn)
    scales = torch.full((1, 1), 255, dtype=torch.uint8, device=device)
    identity = torch.eye(32, device=device, dtype=torch.bfloat16)
    assert mxfp8_w8a16_linear(identity, weight, scales).isnan().all()


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_mxfp8_torch_reference_matches_the_kernel_decode():
    device = torch.device("cuda", torch.cuda.current_device())
    weight, scales, dense = _mxfp8_operands(48, 256, device, seed=37)
    assert torch.equal(dequant_mxfp8(weight, scales).double(), dense)


# --------------------------------------------------------------------------- #
# Dispatch helper
# --------------------------------------------------------------------------- #


def test_scaled_wxa16_available_rejects_non_cuda():
    if TRITON_AVAILABLE is False:
        pytest.skip("Triton is not installed")
    assert scaled_wxa16_available(torch.device("cpu")) is False
    assert scaled_wxa16_available("meta") is False
