# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5772: CPU embedding lookups use row decode.

``TorchQuantEmbeddings`` applied the GPU-measured 64 MiB table minimum and
1024-token maximum to CPU, so a small table or one token above the cutoff
decoded the whole vocabulary before gathering. CPU now decodes only the
requested rows, in bounded chunks, while the accelerator cutoffs stay intact.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT


VOCAB = 4096
DIM = 256


def _module(bits=4, fmt=FORMAT.GPTQ_V2, group_size=32):
    torch.manual_seed(5772)
    embedding = nn.Embedding(VOCAB, DIM)
    groups = math.ceil(VOCAB / group_size)
    module = TorchQuantEmbeddings(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=VOCAB,
        out_features=DIM,
        bias=False,
        format=fmt,
    ).eval()
    module.pack_original(
        embedding,
        scales=(torch.rand(DIM, groups) * 0.02).clamp_min(1e-3),
        zeros=torch.randint(0, 1 << bits, (DIM, groups)),
        g_idx=torch.arange(VOCAB, dtype=torch.int32) // group_size,
    )
    module.optimize = lambda *args, **kwargs: None
    module.post_init()
    return module


def _track_dequant(module):
    """Count whole-table decodes so the chosen lookup path is observable."""

    calls = []
    original = module.dequantize_weight

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    module.dequantize_weight = tracked
    return calls


def test_cpu_small_table_uses_row_decode():
    # 4096 x 256 fp16 is 2 MiB, below the accelerator 64 MiB table minimum.
    module = _module()
    reference = module.dequantize_weight()
    calls = _track_dequant(module)
    input_ids = torch.tensor([[0, 1, 2, VOCAB - 1]])

    result = module(input_ids)

    assert calls == [], "a CPU lookup must not decode the whole table"
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)


def test_cpu_above_token_cutoff_uses_row_decode():
    module = _module()
    reference = module.dequantize_weight()
    calls = _track_dequant(module)
    input_ids = torch.randint(0, VOCAB, (1, module.EMBEDDING_ROW_DECODE_LIMIT + 1))

    result = module(input_ids)

    assert calls == [], "one token above the GPU cutoff must still use row decode on CPU"
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)


def test_cpu_more_lookups_than_vocabulary_rows_uses_full_decode():
    module = _module()
    reference = module.dequantize_weight()
    calls = _track_dequant(module)
    input_ids = torch.randint(0, VOCAB, (2 * VOCAB,))

    result = module(input_ids)

    assert len(calls) == 1, "more lookups than table rows must use one whole-table decode"
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)


@pytest.mark.parametrize("chunk", [1, 7, 1024])
def test_cpu_row_decode_chunks_match_full_decode(chunk):
    module = _module()
    reference = module.dequantize_weight()
    module.EMBEDDING_CPU_ROW_DECODE_CHUNK = chunk
    calls = _track_dequant(module)
    input_ids = torch.randint(0, VOCAB, (3, 200))

    result = module(input_ids)

    assert calls == []
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_row_decode_preserves_output_dtype(dtype):
    module = _module().to(dtype)
    reference = module.dequantize_weight()
    input_ids = torch.randint(0, VOCAB, (2, 64))

    result = module(input_ids)

    assert result.dtype == dtype
    torch.testing.assert_close(result, F.embedding(input_ids, reference), rtol=0, atol=0)


def test_cpu_invalid_ids_raise_on_both_policies():
    module = _module()

    # Row policy: a single lookup is never wider than the table.
    for invalid in (VOCAB, -1):
        with pytest.raises(IndexError):
            module(torch.tensor([invalid]))

    # Whole-table policy: more lookups than table rows, with one invalid id.
    ids = torch.cat([torch.randint(0, VOCAB, (VOCAB,)), torch.tensor([VOCAB])])
    with pytest.raises(IndexError):
        module(ids)
