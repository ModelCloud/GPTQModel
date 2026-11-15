# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Sequential TT block-float GPTQ must replay each block with its own context.

``quantize_model_blockfloat(sequential=True)`` captures the side inputs a decoder
block receives (attention mask, rotary embeddings, position ids, cache
positions) and replays them while quantizing block by block.  Capturing only
block 0 and reusing its kwargs is wrong for hybrid-attention models: Gemma2/3
build a different causal mask per ``attention_type``, so a full-attention layer
would calibrate on the sliding-window mask (regression from #4009, #4150).
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

from gptqmodel.utils.tt_blockfloat_quant import quantize_model_blockfloat


pytestmark = [pytest.mark.cpu, pytest.mark.slow]

_VOCAB = 32
_SEQLEN = 16
_WINDOWS = 2


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FixedTokenizer:
    """Deterministic tokenizer stub: one 40-token stream, ids stay in vocab."""

    def __call__(self, text, return_tensors=None):
        del text, return_tensors
        return _Tokenized(torch.arange(40).unsqueeze(0) % _VOCAB)


def _build_model(layer_types):
    config = Gemma2Config(
        vocab_size=_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=layer_types,
        sliding_window=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Gemma2ForCausalLM(config).eval()


def _native_layer_masks(model):
    masks = {}

    def _record(index):
        def hook(_module, _args, kwargs):
            mask = kwargs.get("attention_mask")
            masks[index] = None if mask is None else mask.detach().clone()

        return hook

    handles = [
        block.register_forward_pre_hook(_record(index), with_kwargs=True)
        for index, block in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            model(torch.arange(_SEQLEN).unsqueeze(0) % _VOCAB)
    finally:
        for handle in handles:
            handle.remove()
    return masks


def _sequential_replay_masks(model):
    masks = {index: [] for index in range(len(model.model.layers))}

    def _record(index):
        def hook(_module, _args, kwargs):
            mask = kwargs.get("attention_mask")
            masks[index].append(None if mask is None else mask.detach().clone())

        return hook

    handles = [
        block.register_forward_pre_hook(_record(index), with_kwargs=True)
        for index, block in enumerate(model.model.layers)
    ]
    try:
        report = quantize_model_blockfloat(
            model,
            _FixedTokenizer(),
            bits=4,
            damp=0.01,
            nsamples=_WINDOWS,
            text="tiny sequential block float calibration sample",
            seqlen=_SEQLEN,
            sequential=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return report, masks


def _assert_each_block_saw_its_own_mask(layer_types):
    model = _build_model(layer_types)
    native = _native_layer_masks(model)
    report, replayed = _sequential_replay_masks(model)

    assert report, "sequential quantization produced no report"
    for index in range(len(model.model.layers)):
        assert replayed[index], f"layer {index} never replayed"
        for mask in replayed[index]:
            assert mask is not None, f"layer {index} replayed without a mask"
            assert torch.equal(mask, native[index]), (
                f"layer {index} ({layer_types[index]}) replayed with another "
                "layer's attention mask"
            )
    return native, replayed


def test_sequential_replay_uses_each_hybrid_layer_mask():
    native, replayed = _assert_each_block_saw_its_own_mask(
        ["sliding_attention", "full_attention"]
    )

    # The two layer types really do produce different masks, so the assertion
    # above is not vacuous, and the old behavior is named explicitly: layer 1
    # must not receive layer 0's sliding-window mask.
    assert not torch.equal(native[0], native[1])
    assert torch.equal(replayed[1][-1], native[1])
    assert not torch.equal(replayed[1][-1], native[0])


def test_sequential_replay_full_attention_context_unchanged():
    native, replayed = _assert_each_block_saw_its_own_mask(
        ["full_attention", "full_attention"]
    )

    # Pure full-attention models have one mask for every layer; per-block
    # capture must reproduce the historical block-0 context exactly.
    assert torch.equal(native[0], native[1])
    for mask in replayed[0] + replayed[1]:
        assert torch.equal(mask, native[0])
