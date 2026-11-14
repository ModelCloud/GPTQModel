# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ solve onto the Tenstorrent block-float grid with exponent blocks along out_features.

A Tenstorrent device stores the weight operand of ``ttnn.linear(x, W^T)`` as
``[in_features, out_features]`` and shares one block-float exponent per 16
values of the *last* dim.  The default ``block_axis="in"`` solve blocks along
in_features, so uploading its result as ``W^T`` re-quantizes it on the host.
``block_axis="out"`` produces weights that are exactly on the grid of ``W^T``.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.utils import tt_blockfloat_quant as bq


def _problem(out_features=64, in_features=128, tokens=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Correlated activations with per-channel scale spread, so error feedback
    # and column ordering both matter.
    x = torch.randn(tokens, 32, generator=g) @ torch.randn(32, in_features, generator=g)
    x = x * torch.logspace(-1, 1, in_features)[torch.randperm(in_features, generator=g)]
    w = torch.randn(out_features, in_features, generator=g) * 0.02
    return x, w, x.T @ x


def _out_err(x, w, wq):
    ref = x @ w.T
    return float((ref - x @ wq.T).abs().sum() / ref.abs().sum())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("act_order", [False, True])
def test_out_axis_solve_is_on_transposed_grid(bits, act_order):
    x, w, h = _problem()
    q = bq.gptq_solve_blockfloat(w, h, bits=bits, block_axis="out", act_order=act_order)
    assert q.shape == w.shape
    # On the grid of W^T (blocks along out_features) ...
    assert bq.is_on_blockfloat_grid(q, bits, block_axis="out")
    qt = q.T.contiguous()
    assert torch.equal(bq.quantize_blockfloat(qt, bits)[0], qt)
    # ... and therefore W^T round-trips exactly through the checkpoint code format.
    codes, exps = bq.decompose_blockfloat(qt, bits=bits)
    assert torch.equal(bq.recompose_blockfloat(codes, exps, bits=bits), qt)
    packed = bq._pack_codes(codes.to(torch.int64), bits)
    assert torch.equal(bq._unpack_codes(packed, bits, tuple(qt.shape)), codes)


@pytest.mark.parametrize("bits", [4, 8])
def test_out_axis_solve_beats_rounding(bits):
    x, w, h = _problem()
    rtn = bq.quantize_blockfloat(w.T.contiguous(), bits)[0].T
    for act_order in (False, True):
        q = bq.gptq_solve_blockfloat(w, h, bits=bits, block_axis="out", act_order=act_order)
        assert _out_err(x, w, q) < _out_err(x, w, rtn)


def test_act_order_helps_out_axis_bf4():
    x, w, h = _problem(seed=3)
    plain = bq.gptq_solve_blockfloat(w, h, bits=4, block_axis="out", act_order=False)
    ordered = bq.gptq_solve_blockfloat(w, h, bits=4, block_axis="out", act_order=True)
    assert _out_err(x, w, ordered) < _out_err(x, w, plain)


def test_in_axis_result_is_not_on_transposed_grid():
    """The historical layout is what forces a device-side re-quantization."""

    x, w, h = _problem()
    q = bq.gptq_solve_blockfloat(w, h, bits=4)
    assert bq.is_on_blockfloat_grid(q, 4, block_axis="in")
    assert not bq.is_on_blockfloat_grid(q, 4, block_axis="out")


def test_lazy_batch_matches_unbatched_update():
    """The lazy-batch tail GEMM is an exact reordering of the rank-1 updates."""

    x, w, h = _problem(out_features=32, in_features=96)
    a = bq._gptq_solve_blockfloat_out(
        w, h, bits=8, damp=0.01, block=16, grid="blockfloat", act_order=False, lazy_block=128
    )
    b = bq._gptq_solve_blockfloat_out(
        w, h, bits=8, damp=0.01, block=16, grid="blockfloat", act_order=False, lazy_block=1
    )
    # Same grid decisions; only float32 summation order differs.
    assert float((a - b).abs().max()) <= float(a.abs().max()) * 2 ** -6


def test_column_quantizer_matches_matrix_quantizer():
    g = torch.Generator().manual_seed(1)
    col = torch.randn(64, generator=g) * torch.logspace(-3, 0, 64)
    for bits in (4, 8):
        expect = bq.quantize_blockfloat(col.reshape(4, 16), bits)[0].reshape(-1)
        assert torch.equal(bq.quantize_blockfloat_column(col, bits), expect)


def test_argument_validation():
    x, w, h = _problem(out_features=32, in_features=32)
    with pytest.raises(ValueError, match="block_axis"):
        bq.gptq_solve_blockfloat(w, h, block_axis="rows")
    with pytest.raises(ValueError, match="act_order"):
        bq.gptq_solve_blockfloat(w, h, act_order=True)  # in-axis + act_order
    with pytest.raises(ValueError, match="out_features"):
        bq.gptq_solve_blockfloat(w[:24], h, block_axis="out")
    with pytest.raises(ValueError, match="grid"):
        bq.gptq_solve_blockfloat(w, h, block_axis="out", grid="uniform")


def test_quantize_model_out_axis_embedding_and_layers():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()

    class _Tok:
        def __call__(self, text, return_tensors=None, **_):
            ids = torch.tensor([[ord(c) % 256 for c in text]])
            return type("E", (), {"input_ids": ids})()

    report = bq.quantize_model_blockfloat(
        model, _Tok(), bits={"*": 4, "o_proj": 8}, nsamples=2,
        text=["the quick brown fox jumps over the lazy dog " * 20] * 4,
        embed_bits=8, block_axis="out", act_order=True,
    )
    assert report and all(v["on_grid"] for v in report.values())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in f".{name}":
            bits = 8 if name.endswith("o_proj") else 4
            assert bq.is_on_blockfloat_grid(module.weight, bits, block_axis="out"), name
    assert bq.is_on_blockfloat_grid(model.model.embed_tokens.weight, 8, block_axis="out")
