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


@pytest.mark.parametrize("cache_type", ["dynamic", "static"])
@pytest.mark.parametrize(
    "past_length,new_length,full_prompt,capacity,visible_tail,explicit_cache",
    [
        pytest.param(4, 2, True, 10, True, False, id="review-visible-capacity"),
        pytest.param(10, 2, False, 16, False, False, id="review-short-padded-block"),
        pytest.param(4, 2, True, 10, False, False, id="zero-exact-capacity"),
        pytest.param(4, 2, True, 12, False, False, id="zero-long-capacity"),
        pytest.param(4, 2, True, 6, True, False, id="full-prompt"),
        pytest.param(4, 4, False, 8, True, False, id="equal-length-block"),
        pytest.param(4, 1, False, 10, False, False, id="single-token-padded"),
        pytest.param(4, 6, False, 10, True, True, id="long-block-explicit-cache"),
        pytest.param(4, 6, False, 16, False, True, id="long-block-padded"),
        pytest.param(4, 2, True, 10, True, True, id="full-prompt-explicit-cache"),
        pytest.param(0, 6, True, 10, False, False, id="prefill-padded-capacity"),
    ],
)
@pytest.mark.parametrize("left_padding", [0, 2])
@torch.inference_mode()
def test_cached_logits_match_full_forward(
    model,
    cache_type,
    past_length,
    new_length,
    full_prompt,
    capacity,
    visible_tail,
    explicit_cache,
    left_padding,
):
    total = past_length + new_length
    tokens = torch.arange(3, 3 + total).unsqueeze(0).repeat(2, 1)
    mask = torch.ones(2, capacity, dtype=torch.long)
    mask[0, :left_padding] = 0
    if not visible_tail:
        mask[:, total:] = 0
    cache = StaticCache(config=model.config, max_cache_len=capacity) if cache_type == "static" else DynamicCache()
    if past_length:
        model(
            input_ids=tokens[:, :past_length],
            attention_mask=mask[:, :past_length],
            position_ids=positions(mask[:, :past_length]),
            past_key_values=cache,
            use_cache=True,
        )
    prepared = model.prepare_inputs_for_generation(
        tokens if full_prompt else tokens[:, past_length:],
        past_key_values=cache,
        attention_mask=mask,
        cache_position=torch.arange(past_length, total) if explicit_cache else None,
        use_cache=True,
    )
    # Execute even on the buggy source to capture the cache and numerical regression.
    prepared["logits_to_keep"] = 0
    actual = model(**prepared).logits
    expected = model(
        input_ids=tokens,
        attention_mask=mask[:, :total],
        position_ids=positions(mask[:, :total]),
        use_cache=False,
    ).logits[:, past_length:]
    error = (actual[:, -new_length:] - expected).abs().max().item()
    detail = (
        f"cache_length={int(cache.get_seq_length())}, "
        f"positions={prepared['position_ids'].tolist()}, max_logit_error={error:.8g}"
    )
    assert int(cache.get_seq_length()) == total, detail
    assert torch.equal(prepared["input_ids"], tokens[:, past_length:]), detail
    assert torch.equal(prepared["cache_position"], torch.arange(past_length, total)), detail
    assert torch.equal(prepared["position_ids"], positions(mask)[:, past_length:total]), detail
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5, msg=detail)


@pytest.mark.parametrize("provided_positions", ["full", "capacity", "new"])
@torch.inference_mode()
def test_explicit_position_ids(model, provided_positions):
    tokens = torch.arange(3, 9).unsqueeze(0)
    cache = model(input_ids=tokens[:, :4], use_cache=True).past_key_values
    # Custom caller positions must be preserved, including when already sliced.
    position_ids = {
        "full": torch.arange(20, 26).unsqueeze(0),
        "capacity": torch.arange(20, 30).unsqueeze(0),
        "new": torch.tensor([[24, 25]]),
    }[provided_positions]
    prepared = model.prepare_inputs_for_generation(
        tokens,
        past_key_values=cache,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        position_ids=position_ids,
    )
    assert prepared["input_ids"].tolist() == [[7, 8]]
    assert prepared["position_ids"].tolist() == [[24, 25]]


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
