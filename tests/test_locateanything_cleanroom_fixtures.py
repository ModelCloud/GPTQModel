# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Black-box checks of the LocateAnything port against the clean-room golden fixtures."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.ports.locateanything import modeling_locateanything
from gptqmodel.models.ports.locateanything.configuration_locateanything import LocateAnythingConfig


FIXTURES = Path(__file__).parent / "fixtures/cleanroom/locateanything"
VARIANTS = ("qwen2_gqa", "qwen2_mqa_tied", "qwen3_bias", "qwen2_sliding")
IMPLEMENTATIONS = ("sdpa", "eager")
ATOL = RTOL = 1e-5


def tiny_config(architecture="Qwen2ForCausalLM", kv_heads=2, **text_overrides):
    """The fixtures' tiny configuration with a chosen decoder family and KV head count."""
    payload = json.loads((FIXTURES / "qwen2_gqa/config.json").read_text())
    text = payload["text_config"]
    text["architectures"] = [architecture]
    text["num_key_value_heads"] = kv_heads
    text["model_type"] = "qwen3" if "Qwen3" in architecture else "qwen2"
    text.update(text_overrides)
    return LocateAnythingConfig(**payload)


def load(variant, implementation):
    model = LocateAnythingQModel.loader.from_pretrained(
        FIXTURES / variant, dtype=torch.float32, attn_implementation=implementation
    )
    return model.eval()


def expected(variant):
    return load_file(FIXTURES / variant / "expected.safetensors")


def close(actual, reference):
    torch.testing.assert_close(actual.detach().float(), reference, atol=ATOL, rtol=RTOL)


@pytest.fixture(scope="module", params=[(v, i) for v in VARIANTS for i in IMPLEMENTATIONS], ids=lambda p: f"{p[0]}-{p[1]}")
def case(request):
    variant, implementation = request.param
    return variant, implementation, load(variant, implementation), expected(variant)


def _set_causal(model, value):
    model.config.text_config.causal_attn = value


@torch.no_grad()
def test_text_and_hidden(case):
    variant, impl, model, E = case
    close(model(input_ids=E["in.text_ids"]).logits, E[f"{impl}.text_logits"])
    hidden = model.language_model.model(input_ids=E["in.text_ids"]).last_hidden_state
    close(hidden, E[f"{impl}.text_hidden_last"])


@torch.no_grad()
def test_mtp_window_rules(case):
    variant, impl, model, E = case
    lm = model.language_model
    close(lm(input_ids=E["in.mtp_ids"], use_cache=True).logits, E[f"{impl}.mtp_cached_logits"])
    close(lm(input_ids=E["in.mtp_ids"], use_cache=False).logits, E[f"{impl}.mtp_uncached_logits"])
    try:
        for causal in (0, 1):
            _set_causal(model, bool(causal))
            close(lm(input_ids=E["in.mtp_ids"], use_cache=True).logits, E[f"{impl}.mtp_cached_causal{causal}_logits"])
            close(
                model(input_ids=E["in.multiwin_ids"], use_cache=False).logits,
                E[f"{impl}.multiwin_uncached_causal{causal}_logits"],
            )
    finally:
        _set_causal(model, False)
    close(
        lm(input_ids=E["in.mtp_gen_ids"], position_ids=E["in.mtp_gen_pos"], use_cache=True).logits,
        E[f"{impl}.mtp_gen_logits"],
    )
    if variant == "qwen3_bias":
        # Native causal attention only: no block-diffusion window rule.
        torch.testing.assert_close(E[f"{impl}.mtp_cached_logits"], E[f"{impl}.mtp_uncached_logits"])
    else:
        assert not torch.allclose(E[f"{impl}.mtp_cached_logits"], E[f"{impl}.mtp_uncached_logits"])


@torch.no_grad()
def test_cache_rounds_and_continuation(case):
    variant, impl, model, E = case
    lm = model.language_model
    cache = lm(input_ids=torch.tensor([[1, 4, 6]]), use_cache=True).past_key_values
    for r in (0, 1):
        seen = int(E[f"in.round{r}_seen"])
        out = lm(
            input_ids=E[f"in.round{r}_full_ids"][:, seen:],
            position_ids=E[f"in.round{r}_full_pos"][:, seen:],
            past_key_values=cache,
            use_cache=True,
        )
        close(out.logits, E[f"{impl}.round{r}_logits"])
        assert out.past_key_values is cache
        cache.crop(-6)
    assert cache.get_seq_length() == 9

    prefix = lm(input_ids=E["in.cont_ids"][:, :3], use_cache=True)
    close(prefix.logits, E[f"{impl}.cont_prefix_logits"])
    suffix = lm(input_ids=E["in.cont_ids"][:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    close(suffix.logits, E[f"{impl}.cont_suffix_logits"])
    assert suffix.past_key_values.get_seq_length() == 5


@torch.no_grad()
def test_left_padded_batch(case):
    variant, impl, model, E = case
    logits = model.language_model(
        input_ids=E["in.batch_ids"], attention_mask=E["in.batch_mask"], position_ids=E["in.batch_pos"]
    ).logits
    reference = E[f"{impl}.batch_logits"]
    # Fully hidden (left-padded) query rows are unspecified; compare real rows only.
    close(logits[1], reference[1])
    if impl == "sdpa":
        close(logits[0, 2:], reference[0, 2:])


@torch.no_grad()
def test_vision_connector_and_multimodal(case):
    variant, impl, model, E = case
    pixels, grids = E["in.pixel_values"], E["in.image_grid_hws"]
    patch = model.vision_model.patch_embed(pixels, grids)
    close(patch, E[f"{impl}.patch_embed_out"])
    close(model.vision_model.encoder(patch, grids), E[f"{impl}.vision_encoder_out"])
    merged = model.extract_feature(pixels, grids)
    assert isinstance(merged, list) and len(merged) == 2
    for i, features in enumerate(merged):
        close(features, E[f"{impl}.vision_merged_{i}"])
    close(model.mlp1(torch.cat(merged)), E[f"{impl}.mlp1_out"])

    close(model(input_ids=E["in.mm_ids"], pixel_values=pixels, image_grid_hws=grids).logits, E[f"{impl}.mm_logits"])
    close(
        model(input_ids=E["in.flag_ids"], pixel_values=pixels, image_grid_hws=grids, image_flags=E["in.image_flags"]).logits,
        E[f"{impl}.flag_logits"],
    )
    flag00 = model(
        input_ids=torch.tensor([[1, 5, 7]]), pixel_values=pixels, image_grid_hws=grids, image_flags=torch.tensor([0, 0])
    ).logits
    close(flag00, E[f"{impl}.flag00_logits"])
    close(flag00, model(input_ids=torch.tensor([[1, 5, 7]])).logits)


def test_training_block_mask_logits_and_loss(case):
    variant, impl, model, E = case
    model.train()
    try:
        for causal in (0, 1):
            _set_causal(model, bool(causal))
            with torch.no_grad():
                out = model(
                    input_ids=E["in.train_ids"], position_ids=E["in.train_pos"], use_cache=False, labels=E["in.train_ids"]
                )
            close(out.logits, E[f"{impl}.train_causal{causal}_logits"])
            close(out.loss, E[f"{impl}.train_causal{causal}_loss"])
    finally:
        _set_causal(model, False)
        model.eval()


@torch.no_grad()
def test_hf_greedy_generate(case):
    variant, impl, model, E = case
    if impl != "sdpa" or variant == "qwen2_sliding":
        pytest.skip("HF greedy ids are recorded for sdpa on non-sliding variants only")
    lm = model.language_model
    ids = lm.generate(torch.tensor([[1, 5, 9, 11, 6]]), max_new_tokens=6, do_sample=False)
    assert torch.equal(ids, E["sdpa.hf_greedy_ids"])
    batch = lm.generate(
        input_ids=E["in.batch_ids"], attention_mask=E["in.batch_mask"], max_new_tokens=4, do_sample=False, pad_token_id=0
    )
    assert torch.equal(batch, E["sdpa.hf_greedy_batch_ids"])


class StubTokenizer:
    model_max_length = 64

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [",".join(str(int(t)) for t in row) for row in sequences]

    def decode(self, tokens, **_kwargs):
        return ",".join(str(int(t)) for t in tokens)


def _rig(model):
    lm = model.language_model
    with torch.no_grad():
        for name, param in lm.named_parameters():
            if "norm" not in name:
                param.zero_()
        lm.model.embed_tokens.weight[:, 0] = 1
        lm.lm_head.weight[32, 0] = 1
    return model


def _run_localization(model, monkeypatch, input_ids, mode, max_new_tokens, **media):
    rounds, calls = [], []

    def recording_handle_pattern(candidate, token_ids, generation_mode="hybrid"):
        result = original(candidate, token_ids, generation_mode=generation_mode)
        rounds.append(
            {"kind": "mtp", "candidate": [int(t) for t in candidate], "type": result["type"], "tokens": list(result["tokens"])}
        )
        return result

    original = modeling_locateanything.handle_pattern
    monkeypatch.setattr(modeling_locateanything, "handle_pattern", recording_handle_pattern)
    hook = model.language_model.register_forward_hook(
        lambda _m, _a, kwargs, out: calls.append([kwargs["input_ids"].shape[1], out.past_key_values.get_seq_length()]),
        with_kwargs=True,
    )
    try:
        response = model.generate(
            input_ids=input_ids,
            tokenizer=StubTokenizer(),
            use_cache=True,
            max_new_tokens=max_new_tokens,
            generation_mode=mode,
            **media,
        )
    finally:
        hook.remove()
        monkeypatch.setattr(modeling_locateanything, "handle_pattern", original)
    return {"response": response, "mtp_rounds": rounds, "forward_calls_[input_len,cache_len_after_forward]": calls}


@pytest.mark.parametrize("variant", VARIANTS)
def test_localization_generate(variant, monkeypatch):
    meta = json.loads((FIXTURES / variant / "expected_meta.json").read_text())["localization_generate"]
    E = expected(variant)
    model = load(variant, "sdpa")
    media = {"pixel_values": E["in.pixel_values"], "image_grid_hws": E["in.image_grid_hws"]}
    for mode in ("slow", "fast", "hybrid"):
        got = _run_localization(model, monkeypatch, torch.tensor([[1, 4, 6]]), mode, 12)
        assert got == meta[f"text_{mode}"], mode
        got = _run_localization(model, monkeypatch, E["in.mm_ids"], mode, 12, **media)
        assert got == meta[f"image_{mode}"], mode
    rigged = _rig(load(variant, "sdpa"))
    for mode in ("slow", "fast", "hybrid"):
        got = _run_localization(rigged, monkeypatch, torch.tensor([[1, 4, 6]]), mode, 8)
        assert got == meta[f"rigged_{mode}"], mode


def test_mask_tables_match_fixture():
    from gptqmodel.models.ports.locateanything.modeling_decoder import LocateAnythingQwen2Model

    tables = load_file(FIXTURES / "masks.safetensors")
    model = LocateAnythingQwen2Model(tiny_config().text_config).eval()

    def visibility(ids, past=0, use_cache=True, attention_mask=None, positions=None, training=False):
        ids = torch.tensor([ids])
        embeds = torch.zeros(1, ids.shape[1], 32)
        if positions is None:
            positions = torch.arange(past, past + ids.shape[1])[None]
        model.train(training)
        mask = model.build_attention_mask(ids, attention_mask, torch.as_tensor(positions), embeds, past, use_cache)
        return (mask[0, 0] == 0).to(torch.uint8)

    window = [1, 4, 5, 6] + [60] * 5
    for causal in (0, 1):
        model.config.causal_attn = bool(causal)
        cases = {
            "cached_window_9": visibility(window),
            "uncached_window_9": visibility(window, use_cache=False),
            "cached_window_past5_len7": visibility([7, 8] + [60] * 5, past=5),
            "uncached_multiwin_17": visibility([1, 4, 6, 9] + [60] * 5 + [8, 9, 10] + [60] * 5, use_cache=False),
            "uncached_leading_mask_8": visibility([60, 60, 60, 5, 6, 7, 60, 60], use_cache=False),
            "train_18": visibility([1] * 18, positions=[list(range(6)) * 2 + list(range(3, 9))], training=True),
            "train_12": visibility([1] * 12, positions=[list(range(5)) + list(range(2, 9))], training=True),
            "train_nodrop_7": visibility([1] * 7, positions=[list(range(7))], training=True),
        }
        for name, table in cases.items():
            assert torch.equal(table, tables[f"{name}_causal{causal}"]), (name, causal)
    model.config.causal_attn = False
    padded = visibility(window, attention_mask=torch.tensor([[0] + [1] * 8]))
    assert torch.equal(padded, tables["cached_window_9_padded_first"])
