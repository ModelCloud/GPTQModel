# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`pack_block` packs a whole input block per shift/reduce, bit-exactly.

The CPU packer used to run one small shift/sum chain per 32-input group and
quantize in the transposed `[out, block]` layout. The block is now quantized in
the packed `[block, out]` layout and every group is packed by a single
vectorized reduce, with the block bounded by an element budget. These tests pin
the packed words to an independent per-group Torch oracle so the batching and
the budget cannot drift the checkpoint bytes.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules import qlinear as qlinear_module
from gptqmodel.nn_modules.qlinear import (
    PACK_BLOCK_MAX_BLOCK_ELEMS,
    _pack_block_block_in,
    _pack_rows_3,
)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.planar_packing import PLANAR_BITS, planar_pack_rows


SUPPORTED_BITS = (2, 3, 4, 8)


def _module(
    bits: int,
    in_features: int,
    out_features: int,
    group_size: int,
    format: FORMAT = FORMAT.GPTQ,
) -> TorchLinear:
    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
        format=format,
    )
    module.qzero_format(format=2)
    return module


def _inputs(bits: int, in_features: int, out_features: int, group_size: int):
    generator = torch.Generator().manual_seed(1234)
    groups = in_features // group_size
    maxq = (1 << bits) - 1

    codes = torch.randint(0, maxq + 1, (in_features, out_features), generator=generator).to(torch.int32)
    scales = (torch.rand(out_features, groups, generator=generator) * 0.02 + 0.005).to(torch.float16)
    zeros = torch.full((out_features, groups), maxq // 2, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    scale_full = scales.t().to(torch.float32)[g_idx.long()]
    zero_full = zeros.t().to(torch.float32)[g_idx.long()]
    weight = (scale_full * (codes.to(torch.float32) - zero_full)).to(torch.float16)

    linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16)
    with torch.no_grad():
        linear.weight.copy_(weight.t())
    return linear, scales, zeros, g_idx, codes


def _oracle_codes(linear, scales, zeros, g_idx, bits):
    """Recompute the quantized codes with plain Torch, outside the packer."""
    maxq = (1 << bits) - 1
    weight = linear.weight.detach().to(torch.float32)  # [out, in]
    scale_full = scales.t().to(torch.float32)[g_idx.long()]  # [in, out]
    zero_full = zeros.t().to(torch.float32)[g_idx.long()]  # [in, out]
    int_weight = weight + (zero_full * scale_full).t()
    int_weight = int_weight.div(scale_full.t())
    return int_weight.round().clamp(0, maxq).to(torch.int32).t().contiguous()  # [in, out]


def _oracle_words(bits: int, int_weight: torch.Tensor) -> torch.Tensor:
    """Independently pack `[in, out]` codes the way the checkpoint stores them."""
    if bits in (2, 4, 8):
        pack_factor = 32 // bits
        rows = int_weight.shape[0] // pack_factor
        shifted = int_weight.reshape(rows, pack_factor, -1).to(torch.int64) << torch.arange(
            pack_factor, dtype=torch.int64
        ).view(1, pack_factor, 1) * bits
        return shifted.sum(dim=1).to(torch.int32)
    # 3-bit continuous layout, written out one group at a time.
    groups = int_weight.shape[0] // 32
    out = torch.zeros((groups * 3, int_weight.shape[1]), dtype=torch.int32)
    for group in range(groups):
        x = int_weight[group * 32:(group + 1) * 32]
        a = torch.zeros_like(x[0])
        for j in range(10):
            a |= x[j] << (3 * j)
        a |= x[10] << 30
        b = (x[10] >> 2) & 1
        for j in range(10):
            b |= x[11 + j] << (3 * j + 1)
        b |= x[21] << 31
        c = (x[21] >> 1) & 0x3
        for j in range(10):
            c |= x[22 + j] << (3 * j + 2)
        out[group * 3 + 0] = a
        out[group * 3 + 1] = b
        out[group * 3 + 2] = c
    return out


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize(
    ("in_features", "out_features", "group_size", "block_in"),
    [
        (128, 64, 128, 8192),
        (256, 128, 128, 32),
        (512, 256, 128, 64),
        (1024, 512, 128, 128),
        (1024, 1024, 128, 8192),
        (2048, 256, 128, 96),
    ],
)
def test_pack_block_matches_independent_oracle(bits, in_features, out_features, group_size, block_in):
    linear, scales, zeros, g_idx, _ = _inputs(bits, in_features, out_features, group_size)

    module = _module(bits, in_features, out_features, group_size)
    module.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        block_in=block_in,
        workers=1,
    )

    int_weight = _oracle_codes(linear, scales, zeros, g_idx, bits)
    expected = _oracle_words(bits, int_weight)
    assert torch.equal(module.qweight, expected)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_block_budget_does_not_change_packed_words(bits: int):
    """The element budget only re-partitions the input axis."""
    in_features, out_features, group_size = 2048, 512, 128
    linear, scales, zeros, g_idx, _ = _inputs(bits, in_features, out_features, group_size)

    reference = None
    for budget in (64 << 10, 256 << 10, 1 << 20, 8 << 20):
        previous = qlinear_module.PACK_BLOCK_MAX_BLOCK_ELEMS
        qlinear_module.PACK_BLOCK_MAX_BLOCK_ELEMS = budget
        try:
            module = _module(bits, in_features, out_features, group_size)
            module.pack_block(
                linear=linear,
                scales=scales.clone(),
                zeros=zeros.clone(),
                g_idx=g_idx.clone(),
                block_in=8192,
                workers=1,
            )
        finally:
            qlinear_module.PACK_BLOCK_MAX_BLOCK_ELEMS = previous
        if reference is None:
            reference = module.qweight
        assert torch.equal(module.qweight, reference), budget


@pytest.mark.parametrize("out_features", [64, 512, 4096, 8960])
def test_block_in_is_word_aligned_and_budgeted(out_features: int):
    for requested in (32, 128, 8192, 1 << 20):
        block_in = _pack_block_block_in(out_features, requested, 32)
        assert block_in % 32 == 0
        assert 32 <= block_in <= requested
        # The live transient is roughly `block_in * out_features` elements, so
        # the cap bounds it (one word is the floor for a very wide layer).
        assert block_in * out_features <= max(PACK_BLOCK_MAX_BLOCK_ELEMS, 32 * out_features)


def test_block_in_never_drops_below_one_word():
    assert _pack_block_block_in(1 << 20, 8192, 32) == 32


def test_batched_3bit_rows_match_the_per_group_schedule():
    generator = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 8, (96, 40), generator=generator, dtype=torch.int32)
    batched = _pack_rows_3(codes)

    # Reference: pack each 32-row group with the original scalar expressions.
    expected = torch.zeros((9, 40), dtype=torch.int32)
    for group in range(3):
        x = codes[group * 32:(group + 1) * 32]
        a = (x[0:10] << (torch.arange(10) * 3).view(-1, 1)).sum(dim=0, dtype=torch.int32) | (x[10] << 30)
        b = ((x[10] >> 2) & 1) | (x[11:21] << (torch.arange(10) * 3 + 1).view(-1, 1)).sum(dim=0, dtype=torch.int32)
        b |= x[21] << 31
        c = ((x[21] >> 1) & 0x3) | (x[22:32] << (torch.arange(10) * 3 + 2).view(-1, 1)).sum(dim=0, dtype=torch.int32)
        expected[group * 3 + 0] = a
        expected[group * 3 + 1] = b
        expected[group * 3 + 2] = c

    assert torch.equal(batched, expected)


def test_batched_3bit_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="32-aligned"):
        _pack_rows_3(torch.zeros((40, 8), dtype=torch.int32))


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_pack_block_saturates_and_keeps_caller_weight(bits: int):
    """Clamping still runs, and the in-place quantization never mutates `linear`."""
    in_features, out_features, group_size = 128, 64, 128
    maxq = (1 << bits) - 1
    linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16)
    with torch.no_grad():
        linear.weight.fill_(1e4)
        linear.weight[:, 0] = -1e4

    scales = torch.full((out_features, in_features // group_size), 0.02, dtype=torch.float16)
    zeros = torch.full((out_features, in_features // group_size), maxq // 2, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    weight_before = linear.weight.detach().clone()
    module = _module(bits, in_features, out_features, group_size)
    module.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        workers=1,
    )

    assert torch.equal(linear.weight, weight_before)
    assert torch.equal(
        module.qweight,
        _oracle_words(bits, _oracle_codes(linear, scales, zeros, g_idx, bits)),
    )


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_pack_block_threaded_matches_single_thread(bits: int):
    in_features, out_features, group_size = 512, 256, 128
    linear, scales, zeros, g_idx, _ = _inputs(bits, in_features, out_features, group_size)

    single = _module(bits, in_features, out_features, group_size)
    single.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=128, workers=1)

    threaded = _module(bits, in_features, out_features, group_size)
    threaded.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=128, workers=4)

    assert torch.equal(single.qweight, threaded.qweight)
    assert torch.equal(single.qzeros, threaded.qzeros)


@pytest.mark.parametrize("bits", PLANAR_BITS)
def test_pack_block_planar_batched_rows(bits: int):
    in_features, out_features, group_size = 256, 128, 128
    linear, scales, zeros, g_idx, _ = _inputs(bits, in_features, out_features, group_size)

    module = _module(bits, in_features, out_features, group_size)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=64, workers=1)

    int_weight = _oracle_codes(linear, scales, zeros, g_idx, bits)
    assert torch.equal(module.qweight, planar_pack_rows(int_weight, bits))


@pytest.mark.parametrize("bits", (2, 3, 4, 8))
def test_pack_block_gptq_p_planar_rows(bits: int):
    """The gptq_p split-plane layout packs the whole block in one pass too."""
    in_features, out_features, group_size = 256, 128, 128
    linear, scales, zeros, g_idx, _ = _inputs(bits, in_features, out_features, group_size)

    module = _module(bits, in_features, out_features, group_size, format=FORMAT.GPTQ_P)
    module.pack_block(linear, scales.clone(), zeros.clone(), g_idx.clone(), block_in=64, workers=1)

    int_weight = _oracle_codes(linear, scales, zeros, g_idx, bits)
    assert torch.equal(module.qweight, planar_pack_rows(int_weight, bits))
