# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native generation regressions moved from Ripple HF PR #11 into GPT-QModel."""

import pytest
import torch
from ricursive_test_utils import make_model
from transformers.cache_utils import DynamicCache, StaticCache


@pytest.fixture(scope="module")
def model():
    return make_model(sliding_window=32)


def positions(mask):
    return (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)


@pytest.mark.parametrize("cache_type", ["dynamic", "hybrid", "static"])
@pytest.mark.parametrize("past_length,new_length", [(4, 2), (10, 2), (4, 4), (4, 1), (4, 6), (0, 6)])
@pytest.mark.parametrize("left_padding", [0, 2])
@torch.inference_mode()
def test_cached_logits_match_full_forward(model, cache_type, past_length, new_length, left_padding):
    total = past_length + new_length
    tokens = torch.arange(3, 3 + total).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, total, dtype=torch.long)
    mask[0, :left_padding] = 0
    cache = {
        "dynamic": lambda: DynamicCache(),
        "hybrid": lambda: DynamicCache(config=model.config),
        "static": lambda: StaticCache(config=model.config, max_cache_len=16),
    }[cache_type]()
    if past_length:
        model(
            input_ids=tokens[:, :past_length],
            attention_mask=mask[:, :past_length],
            position_ids=positions(mask[:, :past_length]),
            past_key_values=cache,
            use_cache=True,
        )
    actual = model(
        input_ids=tokens[:, past_length:],
        attention_mask=mask,
        position_ids=positions(mask)[:, past_length:],
        past_key_values=cache,
        use_cache=True,
    ).logits
    expected = model(
        input_ids=tokens, attention_mask=mask, position_ids=positions(mask), use_cache=False
    ).logits[:, past_length:]
    assert int(cache.get_seq_length()) == total
    valid = mask[:, past_length:].bool()
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-6, rtol=2e-5)
    assert torch.equal(actual[valid].argmax(-1), expected[valid].argmax(-1))


@torch.inference_mode()
def test_generate_left_padded_batch(model):
    tokens = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0, "eos_token_id": None}
    batched = model.generate(tokens, attention_mask=mask, **options)
    for row, length in enumerate([2, 4]):
        unpadded = tokens[row : row + 1, -length:]
        single = model.generate(unpadded, attention_mask=torch.ones_like(unpadded), **options)
        assert torch.equal(batched[row, -3:], single[0, -3:])
