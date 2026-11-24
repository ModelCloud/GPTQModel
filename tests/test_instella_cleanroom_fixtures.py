# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/instella/SPEC.md; see NOTICE.
"""Black-box golden-fixture checks for the clean-room Instella port.

Fixtures live in ``tests/fixtures/cleanroom/instella`` (see its README). Every
model is loaded through the registered definition's public loader.
"""

import json
import warnings
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.ports.instella import InstellaConfig


FIXTURES = Path(__file__).parent / "fixtures" / "cleanroom" / "instella"
VARIANTS = ["gqa_default", "mha_bias", "mqa", "rope_linear", "rope_dynamic", "tied"]
BACKENDS = ["eager", "sdpa"]
COMMON = {"vocab_size": 47, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "max_position_embeddings": 8, "pad_token_id": 1, "bos_token_id": 2, "eos_token_id": 3}


@pytest.fixture(autouse=True)
def _deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture(scope="module")
def inputs():
    return load_file(str(FIXTURES / "inputs.safetensors"))


def _expected(variant):
    return load_file(str(FIXTURES / variant / "expected.safetensors"))


def _load(variant, backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = MODEL_MAP["instella"].loader.from_pretrained(str(FIXTURES / variant), attn_implementation=backend)
    assert type(model).__module__ == "gptqmodel.models.ports.instella.modeling_instella"
    return model.eval()


def _matches_reference(actual, expected):
    # The fixture README's documented fallback for cross-host floating-point
    # reassociation. The bit-exact rerun it quotes was measured on the
    # generator host; this host's torch build reassociates by 1-2 ulp.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def _valid(tensor, mask):
    return tensor[mask.bool()]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_forward_entries(variant, backend, inputs):
    expected = _expected(variant)
    ids, mask, pos = inputs["input_ids"], inputs["attention_mask"], inputs["position_ids"]
    short = ids[:1, :6]
    prefix = f"{backend}."

    model = _load(variant, backend)
    out = model(short, use_cache=False, output_hidden_states=True, output_attentions=backend == "eager")
    _matches_reference(out.logits, expected[prefix + "short.logits"])
    assert len(out.hidden_states) == 3
    for idx, hidden in enumerate(out.hidden_states):
        _matches_reference(hidden, expected[f"{prefix}short.hidden_states.{idx}"])
    if backend == "eager":
        assert len(out.attentions) == 2
        for idx, probs in enumerate(out.attentions):
            _matches_reference(probs, expected[f"{prefix}short.attentions.{idx}"])
            assert torch.all(probs.triu(1) == 0)

    _matches_reference(_load(variant, backend)(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits,
           expected[prefix + "padded.logits"])
    loss = _load(variant, backend)(ids[:1], labels=ids[:1], use_cache=False).loss
    assert loss.dim() == 0 and torch.isfinite(loss)
    _matches_reference(loss.reshape(1), expected[prefix + "loss"].reshape(1))

    model = _load(variant, backend)
    embeds_logits = model(inputs_embeds=model.model.embed_tokens(short), use_cache=False).logits
    _matches_reference(embeds_logits, expected[prefix + "inputs_embeds.logits"])
    _matches_reference(embeds_logits, expected[prefix + "short.logits"])

    keep = _load(variant, backend)(short, use_cache=False, logits_to_keep=1).logits
    _matches_reference(keep, expected[prefix + "logits_to_keep1"])
    torch.testing.assert_close(keep, expected[prefix + "short.logits"][:, -1:], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cache_kind", ["dynamic", "static"])
@torch.no_grad()
def test_cache_entries(variant, backend, cache_kind, inputs):
    expected = _expected(variant)
    ids, mask, pos = inputs["input_ids"], inputs["attention_mask"], inputs["position_ids"]
    model = _load(variant, backend)
    if cache_kind == "dynamic":
        cache = DynamicCache(config=model.config)
    else:
        cache = StaticCache(config=model.config, max_cache_len=16)
    steps = []
    for idx, (start, end) in enumerate([(0, 4), (4, 6), (6, 12)]):
        logits = model(ids[:, start:end], attention_mask=mask[:, :end], position_ids=pos[:, start:end],
                       past_key_values=cache, use_cache=True).logits
        _matches_reference(logits, expected[f"{backend}.cache_{cache_kind}.step{idx}"])
        steps.append(logits)
    if cache_kind == "dynamic":
        for layer in range(2):
            _matches_reference(cache.layers[layer].keys, expected[f"{backend}.cache_dynamic.keys.{layer}"])
            _matches_reference(cache.layers[layer].values, expected[f"{backend}.cache_dynamic.values.{layer}"])
    gap = (_valid(torch.cat(steps, dim=1), mask) - _valid(expected[f"{backend}.padded.logits"], mask)).abs().max()
    if variant == "rope_dynamic":
        # Prefill keys keep their cold rotation, so continuation intentionally drifts from the full forward.
        assert 0.3 < gap.item() < 0.5
    else:
        assert gap.item() <= 3e-6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_generation_entries(variant, backend, inputs):
    expected = _expected(variant)
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    greedy = {"attention_mask": mask[:, :6], "max_new_tokens": 8, "do_sample": False, "eos_token_id": None, "pad_token_id": 1}
    cached = _load(variant, backend).generate(ids[:, :6], use_cache=True, **greedy)
    uncached = _load(variant, backend).generate(ids[:, :6], use_cache=False, **greedy)
    assert torch.equal(cached, expected[f"{backend}.generate_greedy_cached"])
    assert torch.equal(uncached, expected[f"{backend}.generate_greedy_uncached"])
    assert torch.equal(cached, uncached)
    beam = _load(variant, backend).generate(ids[:, :3], num_beams=3, max_new_tokens=4, do_sample=False,
                                            use_cache=True, eos_token_id=None, pad_token_id=1)
    assert torch.equal(beam, expected[f"{backend}.generate_beam3"])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_stateful_rope_entries(variant, backend, inputs):
    expected = _expected(variant)
    ids = inputs["input_ids"]
    model = _load(variant, backend)
    before = model(ids[:1, :6], use_cache=False).logits
    long = model(ids[:1], use_cache=False).logits
    after = model(ids[:1, :6], use_cache=False).logits
    _matches_reference(before, expected[f"{backend}.stateful.short_before"])
    _matches_reference(long, expected[f"{backend}.stateful.long"])
    _matches_reference(after, expected[f"{backend}.stateful.short_after"])
    for layer in range(2):
        _matches_reference(model.model.layers[layer].self_attn.rotary_emb.inv_freq,
               expected[f"{backend}.stateful.inv_freq_after.layer{layer}"])
    if variant == "rope_dynamic":
        assert (after - before).abs().max().item() > 0.01
        base = 1e4 * (2 * 12 / 8 - 1) ** (8 / 6)
        warm = 1.0 / (base ** (torch.arange(0, 8, 2).float() / 8))
        torch.testing.assert_close(model.model.layers[0].self_attn.rotary_emb.inv_freq, warm)
    else:
        _matches_reference(after, before)


@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_rotary_entries(variant):
    expected = _expected(variant)
    rotary = _load(variant, "eager").model.layers[0].self_attn.rotary_emb
    _matches_reference(rotary.inv_freq, expected["rotary.inv_freq_initial"])
    cos, sin = rotary(torch.zeros(1, 2, 8), torch.arange(12).unsqueeze(0))
    _matches_reference(cos, expected["rotary.pos0_11.cos"])
    _matches_reference(sin, expected["rotary.pos0_11.sin"])
    _matches_reference(rotary.inv_freq, expected["rotary.inv_freq_after_pos0_11"])


@pytest.mark.parametrize("variant", VARIANTS)
@torch.no_grad()
def test_eager_matches_sdpa_on_valid_positions(variant, inputs):
    expected = _expected(variant)
    mask = inputs["attention_mask"]
    gap = (_valid(expected["eager.padded.logits"], mask) - _valid(expected["sdpa.padded.logits"], mask)).abs().max()
    assert gap.item() <= 5e-6
    sdpa = _load(variant, "sdpa")(inputs["input_ids"], attention_mask=mask, position_ids=inputs["position_ids"],
                                  use_cache=False).logits
    eager = expected["eager.padded.logits"]
    assert (_valid(sdpa, mask) - _valid(eager, mask)).abs().max().item() <= 5e-6


_CASES = json.loads((FIXTURES / "config_cases.json").read_text())


@pytest.mark.parametrize("name", sorted(_CASES["valid"]))
def test_valid_config_cases(name, tmp_path):
    case = _CASES["valid"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = InstellaConfig(**{**COMMON, **case["kwargs"]})
        assert config.rope_parameters == case["rope_parameters"]
        assert config.rope_theta == case["rope_theta"]
        assert config.num_key_value_heads == case["num_key_value_heads"]
        config.save_pretrained(tmp_path)
        saved = json.loads((tmp_path / "config.json").read_text())
        assert ("rope_scaling" in saved) is case["saved_has_rope_scaling"]
        assert saved["rope_parameters"] == case["saved_rope_parameters"]
        for key, value in case["saved_rope_parameters"].items():
            assert type(saved["rope_parameters"][key]) is type(value)
        reloaded = InstellaConfig.from_pretrained(tmp_path)
    assert reloaded.rope_parameters == case["rope_parameters"]
    assert reloaded.rope_theta == case["rope_theta"]


@pytest.mark.parametrize("name", sorted(_CASES["invalid"]))
def test_invalid_config_cases(name):
    case = _CASES["invalid"][name]
    assert case["error"] == "ValueError"
    with pytest.raises(ValueError) as excinfo:
        InstellaConfig(**{**COMMON, **case["kwargs"]})
    assert str(excinfo.value) == case["message"]
