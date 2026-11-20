# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``pack_original`` must pack in input-axis blocks (#5728).

The whole-matrix form materialized the FP32 weight, two full ``[in, out]``
FP32 gathers (``scale_zeros[g_idx]`` and ``scales[g_idx]``), the FP32 result,
its int32 cast and a transposed contiguous copy. On a 32000x4096 BF16
embedding that peaked at ~1.1 GiB of host RSS for a 250 MiB table.

The packer now computes one input block at a time and writes its rows straight
into ``qweight``. These tests pin both halves of that contract:

* the packed words stay byte-identical to the whole-matrix math, for every
  layout the packer serves (2/3/4/8-bit, planar 3/5/6/7, embeddings, narrow
  storage words, and a non-word-aligned tail), and
* no intermediate reaches the full weight shape.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from gptqmodel.nn_modules import qlinear as _qlinear
from gptqmodel.nn_modules.qlinear import (
    _pack_codes_to_words,
    _pack_rows_3,
    _pack_shifts,
)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.planar_packing import planar_pack_rows


# A pre-#5728 library has no block budget. The fallback keeps this file
# runnable against it, so the equivalence cases can show that the base
# implementation agrees on the words while still failing the memory contract.
PACK_ORIGINAL_MAX_BLOCK_ELEMS = getattr(
    _qlinear, "PACK_ORIGINAL_MAX_BLOCK_ELEMS", 1 << 20
)


def _module(bits, in_f, out_f, gs, fmt, pack_dtype):
    return TorchLinear(
        bits=bits,
        group_size=gs,
        sym=True,
        desc_act=False,
        in_features=in_f,
        out_features=out_f,
        format=fmt,
        compile=False,
        pack_dtype=pack_dtype,
    )


def _affine(out_f, n_groups):
    scales = torch.rand(out_f, n_groups) * 0.01 + 1e-3
    zeros = torch.full((out_f, n_groups), 8.0)
    return scales, zeros


def _g_idx(in_f, gs):
    if gs == -1:
        return torch.zeros(in_f, dtype=torch.int32)
    return (torch.arange(in_f) // gs).to(torch.int32)


def _legacy_codes(W, is_embedding, scale_zeros, scales, g_idx, maxq):
    """The whole-matrix quant math this change replaced, verbatim in spirit."""

    W = W.to(torch.float32)
    if is_embedding:
        codes = W + scale_zeros[g_idx]
        codes = codes / scales[g_idx]
    else:
        codes = W + scale_zeros[g_idx].T
        codes = codes / scales[g_idx].T
    codes = codes.round().clamp(0, maxq).to(torch.int32)
    return codes if is_embedding else codes.T.contiguous()


def _legacy_pack(codes, bits, pack_factor, word_bits, planar, pack_dtype):
    rows = math.ceil(codes.shape[0] * bits / word_bits)
    qweight = torch.empty((rows, codes.shape[1]), dtype=torch.int32)
    if bits in (2, 4, 8):
        if pack_factor == 1:
            qweight = codes.contiguous()
        elif codes.shape[0] % pack_factor == 0:
            qweight[:] = _pack_codes_to_words(
                codes, pack_factor, word_bits, _pack_shifts(bits, pack_factor)
            )
        else:
            qweight = torch.zeros_like(qweight)
            for row in range(qweight.shape[0]):
                for lane in range(
                    min(pack_factor, codes.shape[0] - row * pack_factor)
                ):
                    qweight[row] |= codes[row * pack_factor + lane] << (bits * lane)
    elif planar:
        qweight = planar_pack_rows(codes, bits, pack_dtype=pack_dtype)
    else:
        qweight = _pack_rows_3(codes)
    return qweight.to(pack_dtype)


def _legacy_qweight(module, dense, scales, zeros, g_idx):
    """Independent whole-matrix reference for ``module.qweight``."""

    W = dense.weight.detach()
    is_embedding = isinstance(dense, nn.Embedding)
    scales_t = scales.T.to(torch.float32).contiguous()
    zeros_t = zeros.T.to(torch.float32).contiguous()
    codes = _legacy_codes(
        W, is_embedding, zeros_t * scales_t, scales_t, g_idx, module.maxq
    )
    return _legacy_pack(
        codes,
        int(module.bits),
        int(module.pack_factor),
        int(module.pack_dtype_bits),
        bool(module.planar),
        module.pack_dtype,
    )


def _run(bits, in_f, out_f, gs, fmt=FORMAT.GPTQ, pack_dtype=torch.int32,
         embedding=False, g_idx_override=None):
    torch.manual_seed(5728)
    module = _module(bits, in_f, out_f, gs, fmt, pack_dtype)
    dense = nn.Embedding(in_f, out_f) if embedding else nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        dense.weight.copy_(torch.randn_like(dense.weight.float(), generator=None) * 0.02)
    n_groups = 1 if gs == -1 else in_f // gs
    scales, zeros = _affine(out_f, n_groups)
    g_idx = g_idx_override if g_idx_override is not None else _g_idx(in_f, gs)
    expected = _legacy_qweight(module, dense, scales, zeros, g_idx)
    module.pack_original(dense, scales, zeros, g_idx)
    return module, expected


_LAYOUTS = [
    # (id, bits, in, out, group_size, format, pack_dtype, embedding)
    ("b2", 2, 256, 128, 32, FORMAT.GPTQ, torch.int32, False),
    ("b2_embed", 2, 256, 128, 32, FORMAT.GPTQ, torch.int32, True),
    ("b3", 3, 256, 128, 32, FORMAT.GPTQ, torch.int32, False),
    ("b3_embed", 3, 256, 128, 32, FORMAT.GPTQ, torch.int32, True),
    ("b4", 4, 256, 128, 32, FORMAT.GPTQ, torch.int32, False),
    ("b4_embed", 4, 256, 128, 32, FORMAT.GPTQ, torch.int32, True),
    ("b8", 8, 256, 128, 32, FORMAT.GPTQ, torch.int32, False),
    ("b8_embed", 8, 256, 128, 32, FORMAT.GPTQ, torch.int32, True),
    ("b4_int16", 4, 256, 128, 32, FORMAT.GPTQ, torch.int16, False),
    ("b4_int8", 4, 256, 128, 32, FORMAT.GPTQ, torch.int8, False),
    ("b3_planar", 3, 256, 128, 32, FORMAT.GPTQ_P, torch.int32, False),
    ("b5_planar", 5, 256, 128, 32, FORMAT.GPTQ_P, torch.int32, False),
    ("b6_planar", 6, 256, 128, 32, FORMAT.GPTQ_P, torch.int32, False),
    ("b7_planar", 7, 256, 128, 32, FORMAT.GPTQ_P, torch.int32, False),
    ("b5_planar_embed", 5, 256, 128, 32, FORMAT.GPTQ_P, torch.int32, True),
    # `group_size=-1` is the only way to reach `in % pack_factor != 0`; the
    # final storage word is then only partially filled.
    ("b4_tail", 4, 100, 64, -1, FORMAT.GPTQ, torch.int32, False),
    ("b4_tail_embed", 4, 100, 64, -1, FORMAT.GPTQ, torch.int32, True),
    ("b2_tail", 2, 100, 64, -1, FORMAT.GPTQ, torch.int32, False),
    ("b2_tail_embed", 2, 100, 64, -1, FORMAT.GPTQ, torch.int32, True),
    # Wide enough that the block budget actually splits the matrix.
    ("b4_wide", 4, 4096, 256, 128, FORMAT.GPTQ, torch.int32, False),
    ("b4_wide_embed", 4, 4096, 256, 128, FORMAT.GPTQ, torch.int32, True),
]


@pytest.mark.parametrize(
    "name,bits,in_f,out_f,gs,fmt,pack_dtype,embedding",
    _LAYOUTS,
    ids=[layout[0] for layout in _LAYOUTS],
)
def test_packed_words_match_the_whole_matrix_reference(
    name, bits, in_f, out_f, gs, fmt, pack_dtype, embedding
):
    module, expected = _run(bits, in_f, out_f, gs, fmt, pack_dtype, embedding)

    assert module.qweight.dtype is pack_dtype
    assert module.qweight.shape == expected.shape
    assert torch.equal(module.qweight, expected), (
        f"{name}: packed words diverged from the whole-matrix reference"
    )


def test_block_budget_splits_a_wide_matrix():
    # Guard the guard: a matrix this wide must actually be split, otherwise the
    # memory test below would pass for the wrong reason.
    module, _ = _run(4, 4096, 2048, 128)
    rows = math.ceil(4096 * 4 / 32)
    assert module.qweight.shape == (rows, 2048)
    block = min(
        4096, PACK_ORIGINAL_MAX_BLOCK_ELEMS // 2048 // 8 * 8
    )
    assert 0 < block < 4096


def test_no_intermediate_reaches_the_full_weight_shape():
    in_f, out_f = 4096, 2048
    full = in_f * out_f

    module = _module(4, in_f, out_f, 128, FORMAT.GPTQ, torch.int32)
    dense = nn.Linear(in_f, out_f, bias=False)
    scales, zeros = _affine(out_f, in_f // 128)
    g_idx = _g_idx(in_f, 128)

    # Views and `.data` aliases share the caller's storage, so only a storage
    # first seen inside the packer is a transient the packer allocated.
    known = {
        tensor.untyped_storage().data_ptr()
        for tensor in (dense.weight, scales, zeros, g_idx)
    }
    peak = [0]

    class TrackAllocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            if isinstance(output, torch.Tensor):
                storage = output.untyped_storage()
                ptr = storage.data_ptr()
                if ptr not in known:
                    known.add(ptr)
                    peak[0] = max(peak[0], storage.nbytes())
            return output

    with TrackAllocations():
        module.pack_original(dense, scales, zeros, g_idx)

    # The full FP32 weight alone is `full * 4` bytes; the previous form also
    # held two gathers and the int32 transpose at that size.
    assert peak[0] > 0, "the dispatch mode saw no new storage"
    assert peak[0] < full * 4, (
        f"pack_original allocated {peak[0]} bytes; one FP32 copy of the weight "
        f"is {full * 4} bytes"
    )
    # Every transient is block-sized, so the peak must stay near the budget
    # rather than grow with the matrix.
    assert peak[0] <= 4 * PACK_ORIGINAL_MAX_BLOCK_ELEMS * 4


def test_negative_g_idx_matches_its_positive_equivalent():
    """A negative group id wraps to the end of the group axis, as before."""

    in_f, out_f, gs = 256, 128, 32
    n_groups = in_f // gs
    g_idx = _g_idx(in_f, gs)
    wrapped = g_idx.clone()
    wrapped[::7] = -1
    positive = wrapped.clone()
    positive[positive < 0] += n_groups

    module_a, expected_a = _run(4, in_f, out_f, gs, g_idx_override=wrapped)
    module_b, expected_b = _run(4, in_f, out_f, gs, g_idx_override=positive)

    assert torch.equal(module_a.qweight, expected_a)
    assert torch.equal(module_a.qweight, module_b.qweight)


def test_embedding_and_linear_pack_the_same_codes_for_transposed_weights():
    """Orientation must not change the codes: only the input axis is blocked."""

    in_f, out_f, gs = 512, 128, 64
    torch.manual_seed(11)
    base = torch.randn(in_f, out_f) * 0.02

    scales, zeros = _affine(out_f, in_f // gs)
    g_idx = _g_idx(in_f, gs)

    embed = nn.Embedding(in_f, out_f)
    with torch.no_grad():
        embed.weight.copy_(base)
    embed_module = _module(4, in_f, out_f, gs, FORMAT.GPTQ, torch.int32)
    embed_module.pack_original(embed, scales, zeros, g_idx)

    linear = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        linear.weight.copy_(base.T.contiguous())
    linear_module = _module(4, in_f, out_f, gs, FORMAT.GPTQ, torch.int32)
    linear_module.pack_original(linear, scales, zeros, g_idx)

    assert torch.equal(embed_module.qweight, linear_module.qweight)
    assert torch.equal(embed_module.qzeros, linear_module.qzeros)
