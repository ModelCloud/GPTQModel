# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regressions for the full Ripple modeling review; no checkpoint weights needed."""

import pytest
import torch
import torch.nn.functional as F
import transformers
from ricursive_test_utils import make_model


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
def test_logits_projection_is_bounded_for_every_instance():
    first = make_model()
    second = type(first)(first.config).eval()
    for index, model in enumerate([first, second]):
        projected_shapes = []
        handle = model.lm_head.register_forward_hook(lambda _, inp, out: projected_shapes.append(tuple(out.shape)))
        out = model(torch.tensor([[3, 4, 5, 6]]), use_cache=False, logits_to_keep=1)
        handle.remove()
        assert out.logits.shape == (1, 1, 64)
        assert projected_shapes == [(1, 1, 64)], f"instance {index + 1}: {projected_shapes}"


@torch.inference_mode()
def test_logits_to_keep_with_labeled_tuple_output():
    model = make_model()
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
    # Checkpointed layers recompute through `forward`, so count a submodule call.
    handle = model.model.layers[0].mlp.register_forward_pre_hook(lambda *_: calls.append(1))
    actual = model(ids, labels=ids, use_cache=True)
    assert actual.past_key_values is None
    actual.loss.backward()
    handle.remove()
    assert len(calls) == 2, "The decoder must run once forward and once during recomputation"
    torch.testing.assert_close(actual.loss, expected.loss)
    for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=2e-6, rtol=2e-5, msg=name)


@torch.inference_mode()
def test_requested_attention_weights_match_sdpa_and_mask():
    model = make_model()
    ids = torch.arange(3, 12).unsqueeze(0)
    allowed = torch.ones(1, 1, 9, 9, dtype=torch.bool).tril()
    allowed[..., 1] = False
    expected = model(ids, attention_mask=allowed, use_cache=False)
    actual = model(ids, attention_mask=allowed, use_cache=False, output_attentions=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-5)
    for index, weights in enumerate(actual.attentions):
        assert weights.shape == (1, 4, 9, 9)
        # A caller-provided 4D mask is used as given for every layer type (Transformers 5.23 masking_utils).
        visible = allowed
        assert torch.count_nonzero(weights.masked_select(~visible)) == 0
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))




@torch.inference_mode()
def test_default_return_dict_reads_transformers5_config_field(monkeypatch):
    model = make_model()
    # `use_return_dict` is deprecated in Transformers 5 and warns on every read.
    monkeypatch.setattr(type(model.config), "use_return_dict",
                        property(lambda self: pytest.fail("deprecated use_return_dict was read")))
    ids = torch.tensor([[3, 4, 5]])
    assert not isinstance(model(ids, use_cache=False), tuple)
    model.config.return_dict = False
    assert isinstance(model(ids, use_cache=False), tuple)
    assert isinstance(model.model(ids, use_cache=False), tuple)
