# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the full Ripple modeling review; no checkpoint weights needed."""

import pytest
import torch
import torch.nn.functional as F
import transformers
from ricursive_test_utils import compatibility_module, make_model


@torch.inference_mode()
def test_embeddings_only_generation():
    model = make_model()
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = model.generate(ids, max_new_tokens=3)[:, -3:]
    actual = model.generate(
        inputs_embeds=model.get_input_embeddings()(ids), attention_mask=torch.ones_like(ids), max_new_tokens=3
    )
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_explicit_cache_position_advances_during_generation():
    model = make_model()
    ids = torch.arange(3, 13).unsqueeze(0)
    cache = model(ids[:, :4], use_cache=True).past_key_values
    expected = model.generate(ids, max_new_tokens=3)[:, -3:]
    actual = model.generate(
        ids[:, 4:],
        past_key_values=cache,
        attention_mask=torch.ones_like(ids),
        cache_position=torch.arange(4, 10),
        max_new_tokens=3,
    )
    assert torch.equal(actual[:, -3:], expected)


@torch.inference_mode()
def test_explicit_full_prompt_positions_do_not_repeat_cached_tokens():
    model = make_model()
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = model.generate(ids, max_new_tokens=3)
    actual = model.generate(ids, cache_position=torch.arange(4), max_new_tokens=3, return_dict_in_generate=True)
    assert actual.past_key_values.get_seq_length() == 6
    assert torch.equal(actual.sequences, expected)


@torch.inference_mode()
def test_generation_returns_requested_hidden_states():
    model = make_model()
    out = model.generate(
        torch.tensor([[3, 4, 5]]), max_new_tokens=2, return_dict_in_generate=True, output_hidden_states=True
    )
    assert all(step is not None for step in out.hidden_states), out.hidden_states


@torch.inference_mode()
def test_forward_returns_requested_attentions():
    model = make_model()
    out = model(torch.tensor([[3, 4, 5]]), output_attentions=True, use_cache=False)
    assert all(isinstance(item, torch.Tensor) for item in out.attentions), out.attentions


def test_advertised_gradient_checkpointing_can_be_enabled():
    model = make_model()
    assert model.supports_gradient_checkpointing
    model.gradient_checkpointing_enable()
    assert model.is_gradient_checkpointing


@torch.inference_mode()
def test_fp16_loss_stays_finite():
    model = make_model().half()
    ids = torch.arange(3, 19).unsqueeze(0).repeat(2048, 1)
    output = model(ids, labels=ids, use_cache=False)
    expected = F.cross_entropy(output.logits[:, :-1].float().reshape(-1, 64), ids[:, 1:].reshape(-1))
    assert torch.isfinite(expected)
    assert torch.isfinite(output.loss), f"actual={output.loss.item()}, FP32 CE={expected.item()}"


@torch.inference_mode()
def test_default_cache_bounds_sliding_layers():
    model = make_model()
    out = model(torch.arange(3, 23).unsqueeze(0), use_cache=True)
    cache = out.past_key_values
    assert cache.get_seq_length() == 20
    if transformers.__version__.split(".", 1)[0] == "4":
        # Transformers 4 cannot crop a configured sliding layer after it has
        # seen more tokens than its window, so the default must stay a plain
        # full cache that assisted decoding and rollback can crop.
        assert all(layer.keys.shape[-2] == 20 for layer in cache.layers)
        cache.crop(5)
        assert cache.get_seq_length() == 5
    else:
        assert cache.layers[0].keys.shape[-2] <= model.config.sliding_window
        assert cache.layers[1].keys.shape[-2] == 20


@torch.inference_mode()
def test_assisted_decoding_reuses_default_cache_after_window():
    """A cache returned by a direct forward must survive assisted decoding.

    Constructing an explicit full cache here would hide the Transformers 4
    regression: the model's own default previously built a hybrid cache whose
    sliding layers cannot be cropped once they pass the window.
    """
    model = make_model(
        vocab_size=97,
        num_hidden_layers=5,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
        sliding_window=8,
        eos_token_id=96,
    )
    ids = torch.tensor([[3, 4, 5] * 4])
    cache = model(ids[:, :10], use_cache=True).past_key_values
    assert int(cache.get_seq_length()) == 10
    options = {
        "attention_mask": torch.ones_like(ids),
        "max_new_tokens": 5,
        "do_sample": False,
        "eos_token_id": 96,
        "suppress_tokens": [96],
    }
    expected = model.generate(ids, **options)
    actual = model.generate(ids, past_key_values=cache, prompt_lookup_num_tokens=3, **options)
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_short_mask_matches_full_visible_prefix():
    model = make_model()
    ids = torch.arange(3, 15).unsqueeze(0)
    cache = model(ids[:, :10], use_cache=True).past_key_values
    inputs = model.prepare_inputs_for_generation(
        ids[:, 10:], past_key_values=cache, attention_mask=torch.ones(1, 2, dtype=torch.long)
    )
    assert inputs["position_ids"].tolist() == [[10, 11]]
    expected = model(ids, use_cache=False).logits[:, -1:]
    actual = model(**inputs).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mask_dtype", ["additive", "bool"])
@torch.inference_mode()
def test_4d_causal_mask(mask_dtype):
    model = make_model()
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = model(ids, use_cache=False).logits
    allowed = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    mask = allowed if mask_dtype == "bool" else torch.zeros(1, 1, 4, 4).masked_fill(~allowed, float("-inf"))
    actual = model(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_static_cache_is_trimmed_before_kv_head_expansion(monkeypatch):
    import sys

    from transformers.cache_utils import StaticCache

    model = make_model()
    module = sys.modules[type(model).__module__]
    original = module.repeat_kv
    lengths = []

    def record(states, repeats):
        lengths.append(states.shape[-2])
        return original(states, repeats)

    monkeypatch.setattr(module, "repeat_kv", record)
    model(
        torch.tensor([[3, 4, 5, 6]]),
        past_key_values=StaticCache(config=model.config, max_cache_len=32),
        use_cache=True,
    )
    assert lengths == [4, 4, 4, 4], lengths


@torch.inference_mode()
@pytest.mark.parametrize("legacy", [False, True])
def test_gptq_logits_projection_is_bounded_for_every_instance(legacy):
    _patch_logits_to_keep = compatibility_module()._patch_logits_to_keep
    first = make_model()
    if legacy:
        type(first)._ricursive_tf5_native_fix = "2026-11-05"
    second = type(first)(first.config).eval()
    for index, model in enumerate([first, second]):
        _patch_logits_to_keep(model)
        projected_shapes = []
        handle = model.lm_head.register_forward_hook(lambda _, inp, out: projected_shapes.append(tuple(out.shape)))
        out = model(torch.tensor([[3, 4, 5, 6]]), use_cache=False, logits_to_keep=1)
        handle.remove()
        assert out.logits.shape == (1, 1, 64)
        assert projected_shapes == [(1, 1, 64)], f"instance {index + 1}: {projected_shapes}"


@torch.inference_mode()
@pytest.mark.parametrize("legacy", [False, True])
def test_gptq_logits_to_keep_with_labeled_tuple_output(legacy):
    _patch_logits_to_keep = compatibility_module()._patch_logits_to_keep
    model = make_model()
    if legacy:
        type(model)._ricursive_tf5_native_fix = "2026-11-05"
    _patch_logits_to_keep(model)
    ids = torch.tensor([[3, 4, 5, 6]])
    out = model(ids, labels=ids, use_cache=False, logits_to_keep=1, return_dict=False)
    assert torch.isfinite(out[0])
    assert out[1].shape == (1, 1, 64)


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("use_valres", [False, True])
def test_gradient_checkpointing_recomputes_and_preserves_gradients(use_reentrant, use_valres):
    model = make_model(use_valres=use_valres).train()
    reference = type(model)(model.config).train()
    reference.load_state_dict(model.state_dict())
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    expected = reference(ids, labels=ids, use_cache=False)
    expected.loss.backward()

    model.gradient_checkpointing_enable({"use_reentrant": use_reentrant})
    calls = []
    handle = model.model.layers[0].register_forward_pre_hook(lambda *_: calls.append(1))
    actual = model(ids, labels=ids, use_cache=True)
    assert actual.past_key_values is None
    actual.loss.backward()
    handle.remove()
    assert len(calls) == 2, "The decoder must run once forward and once during recomputation"
    torch.testing.assert_close(actual.loss, expected.loss)
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=2e-6, rtol=2e-5, msg=name)


@pytest.mark.parametrize("layout", ["full", "new", "embeddings"])
@torch.inference_mode()
def test_compatibility_generation_advances_positions(layout):
    apply_ricursive_generate_patch = compatibility_module().apply_ricursive_generate_patch
    model = make_model()
    # A historical native marker must not suppress newer generation fixes.
    type(model)._ricursive_tf5_native_fix = "2026-11-05"
    assert apply_ricursive_generate_patch(model)
    ids = torch.arange(3, 13).unsqueeze(0)
    expected = model.generate(ids, max_new_tokens=3)[:, -3:]
    if layout == "embeddings":
        actual = model.generate(
            inputs_embeds=model.get_input_embeddings()(ids), attention_mask=torch.ones_like(ids), max_new_tokens=3
        )
    else:
        cache = model(ids[:, :4], use_cache=True).past_key_values
        actual = model.generate(
            ids if layout == "full" else ids[:, 4:],
            past_key_values=cache,
            cache_position=torch.arange(4, 10),
            attention_mask=torch.ones_like(ids),
            max_new_tokens=3,
        )
    assert torch.equal(actual[:, -3:], expected)


@pytest.mark.parametrize("legacy", [False, True])
@torch.inference_mode()
def test_requested_attention_weights_match_sdpa_and_mask(legacy):
    model = make_model()
    if legacy:
        apply_ricursive_generate_patch = compatibility_module().apply_ricursive_generate_patch
        type(model)._ricursive_tf5_native_fix = "2026-11-05"
        apply_ricursive_generate_patch(model)
    ids = torch.arange(3, 12).unsqueeze(0)
    allowed = torch.ones(1, 1, 9, 9, dtype=torch.bool).tril()
    allowed[..., 1] = False
    expected = model(ids, attention_mask=allowed, use_cache=False)
    actual = model(ids, attention_mask=allowed, use_cache=False, output_attentions=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
    for index, weights in enumerate(actual.attentions):
        assert weights.shape == (1, 4, 9, 9)
        visible = allowed.clone()
        if index == 0:
            positions = torch.arange(9)
            visible &= (positions[:, None] - positions[None, :]) < model.config.sliding_window
        assert torch.count_nonzero(weights.masked_select(~visible)) == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


@torch.inference_mode()
def test_legacy_logits_hook_is_removed_on_exception_and_head_replacement():
    _patch_logits_to_keep = compatibility_module()._patch_logits_to_keep
    model = make_model()
    type(model)._ricursive_tf5_native_fix = "2026-11-05"
    _patch_logits_to_keep(model)
    head = model.lm_head
    hook_count = len(head._forward_pre_hooks)
    with pytest.raises(IndexError):
        model(torch.tensor([[1000]]), use_cache=False, logits_to_keep=1)
    assert len(head._forward_pre_hooks) == hook_count
    model.lm_head = torch.nn.Linear(32, 64, bias=False)
    shapes = []
    handle = model.lm_head.register_forward_hook(lambda _, inp, out: shapes.append(tuple(out.shape)))
    model(torch.tensor([[3, 4, 5]]), use_cache=False, logits_to_keep=1)
    handle.remove()
    assert shapes == [(1, 1, 64)]
    assert not model.lm_head._forward_pre_hooks
