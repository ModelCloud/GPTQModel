# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/mobilellm/SPEC.md; see NOTICE.
"""Golden MobileLLM fixtures (tests/fixtures/cleanroom/mobilellm) through the definition's public loader."""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from gptqmodel.models.auto import MODEL_MAP


ROOT = Path(__file__).parent / "fixtures" / "cleanroom" / "mobilellm"
FIXTURES = sorted(path.name for path in ROOT.iterdir() if (path / "io.safetensors").exists())
FP32 = {"atol": 5e-5, "rtol": 1e-5}
HIDDEN = {"atol": 1e-4, "rtol": 1e-5}


def _load(name, implementation, dtype=torch.float32):
    loader = MODEL_MAP["mobilellm"].loader
    return loader.from_pretrained(ROOT / name, attn_implementation=implementation, dtype=dtype).eval()


def _close(actual, expected, **tolerance):
    torch.testing.assert_close(actual.float(), expected, **tolerance)


def test_fixture_set_is_complete():
    assert FIXTURES == ["base_gqa", "lmhead_tied", "rope_dynamic_sharing", "rope_linear",
                        "sharing_lmhead_bias_mqa", "sharing_mha_shared_emb", "tp2_bias_sharing", "tp2_shared_emb"]


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("name", FIXTURES)
@torch.inference_mode()
def test_cleanroom_fixture(name, implementation):
    data = load_file(ROOT / name / "io.safetensors")
    prefix = implementation + "."

    def expected(key):
        return data[prefix + key]

    ids = data["input_ids"]
    pad_ids, pad_mask, pad_positions = data["pad_input_ids"], data["pad_attention_mask"], data["pad_position_ids"]
    valid = pad_mask.bool()

    model = _load(name, implementation)
    config = model.config
    uses = config.num_hidden_layers * (2 if config.layer_sharing else 1)

    with safe_open(ROOT / name / "model.safetensors", "pt") as handle:
        checkpoint_keys = set(handle.keys())
    state_keys = set(model.state_dict())
    if config.share_embedding or not config.tie_word_embeddings:
        assert checkpoint_keys == state_keys
    else:
        assert checkpoint_keys == state_keys - {"lm_head.weight"}
        assert model.get_output_embeddings().weight is model.model.embed_tokens.weight

    eager = implementation == "eager"
    out = model(ids, use_cache=False, output_hidden_states=True, output_attentions=eager)
    _close(out.logits, expected("logits"), **FP32)
    assert len(out.hidden_states) == uses + 1
    for index, hidden in enumerate(out.hidden_states):
        _close(hidden, expected(f"hidden_states.{index}"), **HIDDEN)
    if eager:
        assert len(out.attentions) == uses
        for index, weights in enumerate(out.attentions):
            _close(weights, expected(f"attentions.{index}"), **FP32)

    pad_logits = model(pad_ids, attention_mask=pad_mask, use_cache=False).logits
    _close(pad_logits[valid], expected("pad_logits")[valid], **FP32)

    cache, chunks, offset = None, [], 0
    for count in (1, 4, 2):
        step = model(ids[:, offset:offset + count], past_key_values=cache, use_cache=True)
        cache, offset = step.past_key_values, offset + count
        chunks.append(step.logits)
    _close(torch.cat(chunks, 1), expected("cached_chunk_logits"), **FP32)
    assert len(cache.layers) == int(expected("cache_num_slots")) == uses
    for slot, layer in enumerate(cache.layers):
        assert layer.get_seq_length() == ids.shape[1]
        _close(layer.keys, expected(f"cache.{slot}.keys"), **FP32)
        _close(layer.values, expected(f"cache.{slot}.values"), **FP32)

    _close(model(ids[:, 1:3], use_cache=False).logits, expected("after_long_short_logits"), **FP32)
    _close(_load(name, implementation)(ids[:, 1:3], use_cache=False).logits, expected("fresh_short_logits"), **FP32)

    fresh = _load(name, implementation)
    cache, chunks, offset = None, [], 0
    for count in (2, 2, 1):
        end = offset + count
        step = fresh(pad_ids[:, offset:end], attention_mask=pad_mask[:, :end],
                     position_ids=pad_positions[:, offset:end], past_key_values=cache, use_cache=True)
        cache, offset = step.past_key_values, end
        chunks.append(step.logits)
    _close(torch.cat(chunks, 1)[valid], expected("pad_cached_logits")[valid], **FP32)

    _close(model(ids, labels=ids).loss, expected("loss"), **FP32)
    _close(model(ids, logits_to_keep=2).logits, expected("logits_to_keep2"), **FP32)
    _close(model(inputs_embeds=model.get_input_embeddings()(ids)).logits, expected("embeds_logits"), **FP32)

    greedy = _load(name, implementation).generate(ids[:, :3], max_new_tokens=5, do_sample=False,
                                                  eos_token_id=None, pad_token_id=0)
    assert torch.equal(greedy, expected("greedy"))
    beam = _load(name, implementation).generate(ids[:1, :3], max_new_tokens=3, num_beams=2, do_sample=False,
                                                eos_token_id=None, pad_token_id=0)
    assert torch.equal(beam, expected("beam2"))

    if eager:
        low = _load(name, implementation, dtype=torch.bfloat16)
        logits = low(ids, use_cache=False).logits
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, expected("bf16_logits"), atol=5e-2, rtol=2e-2)
