# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real tiny native MLA, legacy variants, loading and quantization consumers."""

import copy
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import DeepseekV2ForCausalLM as NativeModel
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache, DynamicLayer, StaticCache
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2RotaryEmbedding as NativeRotaryEmbedding,
)

from gptqmodel import GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from gptqmodel.models.ports.deepseek_v2.modeling_deepseek_v2 import DeepseekV2ForCausalLM
from gptqmodel.models.ports.native.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MLP
from gptqmodel.quantization.config import AWQConfig, QuantizeConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV
from tests.module_tree.test_deepseek_v2_local_port import tiny_config


def config(**overrides):
    values = tiny_config().to_dict()
    values.update(overrides)
    return DeepseekV2Config(**values)


def model(**overrides):
    torch.manual_seed(271)
    return DeepseekV2ForCausalLM(config(**overrides)).eval()


def save_checkpoint(path, hf_model, malicious=False):
    path.mkdir(exist_ok=True)
    hf_model.save_pretrained(path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 3, "[PAD]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    if malicious:
        payload = json.loads((path / "config.json").read_text())
        payload["auto_map"] = {"AutoConfig": "evil.EvilConfig", "AutoModelForCausalLM": "evil.EvilModel"}
        (path / "config.json").write_text(json.dumps(payload))
        (path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')\n")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_tiny_cpu_experts_after_native_model_construction(dtype):
    # Native initialization can prime the inherited capability cache before
    # the local subclass is constructed. This fixture has 12-wide experts,
    # outside grouped_mm's low-precision 16-byte stride requirement.
    native = NativeModel(tiny_config()).eval().to(dtype)
    native.set_experts_implementation("eager")
    actual = model().to(dtype)
    actual.load_state_dict(native.state_dict())
    ids = torch.tensor([[1, 4, 7]])
    torch.testing.assert_close(
        actual(ids, use_cache=False).logits,
        native(ids, use_cache=False).logits,
        rtol=0, atol=0,
    )
    assert actual.config._experts_implementation == "eager"


@pytest.mark.parametrize("q_rank", [None, 8], ids=["lite", "full"])
@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
@torch.inference_mode()
def test_native_numerics_conversion_and_compressed_cache(q_rank, method):
    actual = model(q_lora_rank=q_rank, topk_method=method, routed_scaling_factor=2.3)
    native = NativeModel(tiny_config(q_lora_rank=q_rank, topk_method=method, routed_scaling_factor=2.3)).eval()
    assert native.load_state_dict(actual.state_dict()).missing_keys == []
    ids = torch.tensor([[1, 4, 7, 9, 12], [0, 1, 8, 6, 5]])
    mask = (ids != 0).long()
    expected = native(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual(ids, attention_mask=mask, use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    fused = copy.deepcopy(actual.state_dict())
    assert _convert_model_with_defuser(DeepSeekV2QModel, actual, cleanup_original=True)
    # Every fused expert is split at the declared gate/up boundary, without
    # transpose, loss, or an extra retained fused parameter.
    for i in range(actual.config.n_routed_experts):
        for name, tensor in (
            ("gate_proj", fused["model.layers.1.mlp.experts.gate_up_proj"][i, :12]),
            ("up_proj", fused["model.layers.1.mlp.experts.gate_up_proj"][i, 12:]),
            ("down_proj", fused["model.layers.1.mlp.experts.down_proj"][i]),
        ):
            torch.testing.assert_close(actual.state_dict()[f"model.layers.1.mlp.experts.{i}.{name}.weight"], tensor, rtol=0, atol=0)
    assert not any(k.endswith(("experts.gate_up_proj", "experts.down_proj")) for k in actual.state_dict())
    torch.testing.assert_close(actual(ids, attention_mask=mask, use_cache=False).logits, expected, rtol=2e-5, atol=2e-6)
    prefix = actual(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    cache = prefix.past_key_values
    assert cache.get_seq_length() == 3
    assert cache.layers[0].keys.shape[-1] == actual.config.kv_lora_rank
    assert cache.layers[0].values.shape[-1] == actual.config.qk_rope_head_dim
    continuation = actual(ids[:, 3:], attention_mask=mask, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 5
    assert continuation.past_key_values is cache
    assert continuation.past_key_values.get_seq_length() == 5
    torch.testing.assert_close(continuation.logits, expected[:, 3:], rtol=2e-5, atol=2e-6)
    changed = ids.clone()
    changed[:, -1] = 19
    torch.testing.assert_close(actual(changed, attention_mask=mask, use_cache=False).logits[:, :-1], expected[:, :-1], rtol=2e-5, atol=2e-6)
    generated = actual.generate(ids[:1, :3], max_new_tokens=3, do_sample=False)
    no_cache = actual.generate(ids[:1, :3], max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(generated, no_cache)


@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("top_k", [1, 2])
@torch.inference_mode()
def test_routing_matches_independent_legacy_oracle(method, normalize, top_k):
    hf_model = model(topk_method=method, norm_topk_prob=normalize, num_experts_per_tok=top_k, routed_scaling_factor=3.7)
    router = hf_model.model.layers[1].mlp.gate
    x = torch.randn(2, 5, 24)
    logits = x.reshape(-1, 24).float() @ router.weight.float().T
    scores = logits.softmax(-1)
    candidate = scores.clone()
    if method == "group_limited_greedy":
        group_max = scores.reshape(-1, 2, 2).amax(-1)
        chosen = group_max.argmax(-1)
        allowed = torch.arange(4).div(2, rounding_mode="floor")[None] == chosen[:, None]
        candidate[~allowed] = 0
    indices = candidate.argsort(dim=-1, descending=True)[:, :top_k]
    expected = scores.gather(1, indices)
    if top_k > 1 and normalize:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    else:
        expected *= 3.7
    got_logits, weights, got_indices = router(x)
    order = weights.argsort(-1, descending=True)
    torch.testing.assert_close(got_logits, logits, rtol=0, atol=0)
    assert torch.equal(got_indices.gather(1, order), indices)
    torch.testing.assert_close(weights.gather(1, order), expected)


@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
@pytest.mark.parametrize("scale", [1.0, 1e40, 1e-46])
@torch.inference_mode()
def test_normalized_routing_ignores_even_extreme_finite_scale(method, scale):
    hf_model = model(topk_method=method, norm_topk_prob=True, routed_scaling_factor=scale)
    router = hf_model.model.layers[1].mlp.gate
    x = torch.randn(2, 5, 24)
    logits = x.flatten(0, 1).float() @ router.weight.float().T
    probabilities = logits.softmax(-1)
    candidates = probabilities.clone()
    if method == "group_limited_greedy":
        selected_group = probabilities.unflatten(-1, (2, 2)).amax(-1).argmax(-1)
        allowed = torch.arange(4)[None] // 2 == selected_group[:, None]
        candidates[~allowed] = 0
    indices = candidates.argsort(-1, descending=True)[:, :2]
    expected = candidates.gather(1, indices)
    expected /= expected.sum(-1, keepdim=True) + 1e-20
    got_logits, weights, got_indices = router(x)
    order = weights.argsort(-1, descending=True)
    torch.testing.assert_close(got_logits, logits, rtol=0, atol=0)
    assert torch.isfinite(weights).all()
    assert torch.equal(got_indices.gather(1, order), indices)
    torch.testing.assert_close(weights.gather(1, order), expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 7, 9, 12]])
    reference = model(topk_method=method, norm_topk_prob=True, routed_scaling_factor=1.0)
    actual = hf_model(ids, use_cache=False).logits
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference(ids, use_cache=False).logits, rtol=0, atol=0)


@pytest.mark.parametrize("return_dict,stage,q_rank,dtype,routed,freq", [
    (False, "decoder0", None, torch.float32, None, 1),
    (True, "decoder1", 8, torch.float16, 4, 1),
    (False, "head", None, torch.bfloat16, 4, 2),
    (True, "loss", 8, torch.float32, 4, 2),
    (True, "decoder0", None, torch.float32, 4, 1),
    (False, "decoder1", 8, torch.float32, None, 1),
    (True, "loss", None, torch.float32, 4, 2),
])
@torch.inference_mode()
def test_public_cache_failure_retry_is_transactional(monkeypatch, return_dict, stage, q_rank, dtype, routed, freq):
    hf_model = model(
        q_lora_rank=q_rank, num_hidden_layers=3, n_routed_experts=routed,
        n_shared_experts=2 if routed else None, moe_layer_freq=freq, initializer_range=0.2,
    ).to(dtype)
    ids = torch.tensor([[1, 4, 7, 9, 12]])
    prefix = hf_model(ids[:, :2], use_cache=True).past_key_values
    layers = prefix.layers
    metadata = [(layer, vars(layer).copy()) for layer in layers]
    before = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    observed = []

    def fail(*args, **kwargs):
        observed.append([layer.get_seq_length() for layer in prefix.layers])
        raise RuntimeError("injected after cache update")

    kwargs = {"labels": ids[:, 2:]} if stage == "loss" else {}
    with monkeypatch.context() as patch:
        handle = None
        if stage == "loss":
            patch.setattr(hf_model, "loss_function", fail)
        else:
            module = hf_model.lm_head if stage == "head" else hf_model.model.layers[int(stage[-1])]
            handle = module.register_forward_hook(fail)
        try:
            with pytest.raises(RuntimeError, match="injected after cache update"):
                hf_model(ids[:, 2:], past_key_values=prefix, use_cache=True, return_dict=return_dict, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
    updated = int(stage[-1]) + 1 if stage.startswith("decoder") else 3
    assert observed == [[5] * updated + [2] * (3 - updated)]
    assert prefix.layers is layers
    assert [layer.get_seq_length() for layer in prefix.layers] == [2, 2, 2]
    for layer, (original, state) in zip(prefix.layers, metadata):
        assert layer is original
        assert vars(layer).keys() == state.keys()
        assert all(vars(layer)[key] is value for key, value in state.items())
    for layer, (keys, values) in zip(prefix.layers, before):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)

    retry = hf_model(ids[:, 2:], past_key_values=prefix, use_cache=True, return_dict=return_dict, **kwargs)
    logits = retry.logits if return_dict else retry[1 if kwargs else 0]
    cache = retry.past_key_values if return_dict else retry[2 if kwargs else 1]
    assert type(cache) is DynamicCache
    assert cache is prefix
    assert cache.layers is layers
    assert [layer.get_seq_length() for layer in cache.layers] == [5, 5, 5]
    assert prefix.get_seq_length() == 5
    full = hf_model(ids, use_cache=True)
    rtol, atol = {torch.float32: (2e-5, 2e-6), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(logits, full.logits[:, 2:], rtol=rtol, atol=atol)
    for layer, expected in zip(cache.layers, full.past_key_values.layers):
        torch.testing.assert_close(layer.keys, expected.keys, rtol=rtol, atol=atol)
        torch.testing.assert_close(layer.values, expected.values, rtol=rtol, atol=atol)


@pytest.mark.parametrize("return_dict,q_rank,dtype,routed,freq,native_dense", [
    (False, None, torch.float32, None, 1, 3),
    (True, 8, torch.float16, 4, 1, 1),
    (True, None, torch.bfloat16, 4, 2, 2),
    (False, 8, torch.float32, 4, 2, 2),
])
@torch.inference_mode()
def test_external_cache_manual_1_4_1_native_identity_and_reload(tmp_path, return_dict, q_rank, dtype, routed, freq, native_dense):
    hf_model = model(
        q_lora_rank=q_rank, num_hidden_layers=3, n_routed_experts=routed,
        n_shared_experts=2 if routed else None, moe_layer_freq=freq, norm_topk_prob=False,
        initializer_range=0.2,
    ).to(dtype)
    # Independent native layout: for three layers, first dense 3/1/2 gives
    # respectively all-dense / dense+MoE+MoE / dense+dense+MoE.
    native = NativeModel._from_config(tiny_config(
        q_lora_rank=q_rank, num_hidden_layers=3, first_k_dense_replace=native_dense,
        norm_topk_prob=False, initializer_range=0.2,
    ), experts_implementation="eager").eval().to(dtype)
    native.load_state_dict(hf_model.state_dict(), strict=True)
    native.model.rotary_emb = NativeRotaryEmbedding(config=native.config)
    ids = torch.tensor([[1, 4, 7, 9, 12, 6], [1, 8, 5, 3, 11, 4]])

    def manual(consumer):
        # Intentionally reuse the caller-owned object without assigning outputs
        # back to it: this is HF's mutable-cache public contract.
        external = DynamicCache()
        layers = external.layers
        chunks = []
        for start, end in ((0, 1), (1, 5), (5, 6)):
            output = consumer(ids[:, start:end], past_key_values=external, return_dict=return_dict)
            logits, published = (output.logits, output.past_key_values) if return_dict else output[:2]
            assert published is external
            assert external.layers is layers
            assert [layer.get_seq_length() for layer in external.layers] == [end] * 3
            chunks.append(logits)
        return torch.cat(chunks, dim=1), external

    expected, native_cache = manual(native)
    actual, cache = manual(hf_model)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer, reference in zip(cache.layers, native_cache.layers):
        torch.testing.assert_close(layer.keys, reference.keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, reference.values, rtol=0, atol=0)
    rtol, atol = {torch.float32: (2e-5, 2e-6), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}[dtype]
    torch.testing.assert_close(actual, native(ids, use_cache=False).logits, rtol=rtol, atol=atol)
    hf_model.save_pretrained(tmp_path, safe_serialization=True)
    restored = DeepseekV2ForCausalLM.from_pretrained(tmp_path, dtype=dtype, experts_implementation="eager").eval()
    assert set(restored.state_dict()) == set(hf_model.state_dict())
    for name, tensor in hf_model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    reloaded, reloaded_cache = manual(restored)
    torch.testing.assert_close(reloaded, expected, rtol=0, atol=0)
    for layer, reference in zip(reloaded_cache.layers, native_cache.layers):
        torch.testing.assert_close(layer.keys, reference.keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, reference.values, rtol=0, atol=0)


@pytest.mark.parametrize("return_dict,kind,stage", [
    (False, "lazy", "decoder"), (True, "configured", "lm_decoder"), (False, "reset", "head"),
    (True, "lazy", "loss"), (True, "reset", "decoder"), (False, "configured", "loss"),
])
@torch.inference_mode()
def test_empty_cache_failure_restores_initialization_and_retry(monkeypatch, return_dict, kind, stage):
    hf_model = model(initializer_range=0.2)
    consumer = hf_model.model if stage == "decoder" else hf_model
    ids = torch.tensor([[1, 4, 7]])
    cache = DynamicCache() if kind == "lazy" else DynamicCache(config=hf_model.config)
    if kind == "reset":
        hf_model(ids[:, :1], past_key_values=cache, use_cache=True)
        cache.reset()
    layers = cache.layers
    states = [(layer, vars(layer).copy()) for layer in layers]
    observed = []

    def fail(*args, **kwargs):
        observed.append(cache.get_seq_length())
        raise RuntimeError("injected empty cache failure")

    kwargs = {"labels": ids} if stage == "loss" else {}
    with monkeypatch.context() as patch:
        handle = None
        if stage == "loss":
            patch.setattr(hf_model, "loss_function", fail)
        else:
            target = hf_model.lm_head if stage == "head" else hf_model.model.layers[0]
            handle = target.register_forward_hook(fail)
        try:
            with pytest.raises(RuntimeError, match="injected empty cache failure"):
                consumer(ids, past_key_values=cache, use_cache=True, return_dict=return_dict, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
    assert observed == [3]
    assert cache.layers is layers
    assert len(cache.layers) == len(states)
    for layer, (original, state) in zip(cache.layers, states):
        assert layer is original
        assert vars(layer).keys() == state.keys()
        assert all(vars(layer)[key] is value for key, value in state.items())
    assert cache.get_seq_length() == 0
    retry = consumer(ids, past_key_values=cache, use_cache=True, return_dict=return_dict, **kwargs)
    published = retry.past_key_values if return_dict else retry[2 if kwargs else 1]
    assert published is cache
    assert [layer.get_seq_length() for layer in cache.layers] == [3, 3]
    reference = consumer(ids, use_cache=True)
    torch.testing.assert_close(retry[1 if kwargs else 0], reference[0], rtol=0, atol=0)
    for layer, expected in zip(cache.layers, reference.past_key_values.layers):
        torch.testing.assert_close(layer.keys, expected.keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, expected.values, rtol=0, atol=0)


@torch.inference_mode()
def test_tuple_conversion_failure_rolls_back_public_cache(monkeypatch):
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast

    hf_model = model(initializer_range=0.2)
    ids = torch.tensor([[1, 4, 7, 9, 12]])
    cache = hf_model(ids[:, :2], use_cache=True).past_key_values
    before = [(layer.keys, layer.values) for layer in cache.layers]

    def fail(*args, **kwargs):
        assert [layer.get_seq_length() for layer in cache.layers] == [5, 5]
        raise RuntimeError("injected tuple conversion failure")

    with monkeypatch.context() as patch:
        patch.setattr(MoeCausalLMOutputWithPast, "to_tuple", fail)
        with pytest.raises(RuntimeError, match="injected tuple conversion failure"):
            hf_model(ids[:, 2:], past_key_values=cache, use_cache=True, return_dict=False)
    assert [layer.get_seq_length() for layer in cache.layers] == [2, 2]
    assert all(layer.keys is keys and layer.values is values for layer, (keys, values) in zip(cache.layers, before))
    retry = hf_model(ids[:, 2:], past_key_values=cache, use_cache=True, return_dict=False)
    assert retry[1] is cache
    torch.testing.assert_close(retry[0], hf_model(ids, use_cache=False).logits[:, 2:], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("return_dict", [False, True])
@torch.inference_mode()
def test_native_cache_conversion_methods_and_reset_replay(return_dict):
    hf_model = model()
    ids = torch.tensor([[1, 4, 7, 9, 12], [1, 3, 8, 5, 6]])
    prefix = hf_model(ids[:, :3], use_cache=True).past_key_values
    # Transformers' actual native DDP cache conversion, not a replacement cache.
    cache = DynamicCache(ddp_cache_data=list(prefix))
    assert all(type(layer) is DynamicLayer for layer in cache.layers)
    cache.batch_repeat_interleave(2)
    cache.reorder_cache(torch.tensor([3, 1, 2, 0]))
    cache.batch_select_indices(torch.tensor([0, 1]))
    cache.crop(-1)
    reordered = ids.flip(0)
    result = hf_model(reordered[:, 2:], past_key_values=cache, use_cache=True, return_dict=return_dict)
    logits = result.logits if return_dict else result[0]
    continued = result.past_key_values if return_dict else result[1]
    torch.testing.assert_close(logits, hf_model(reordered, use_cache=False).logits[:, 2:], rtol=2e-5, atol=2e-6)
    assert cache.get_seq_length() == 5
    assert continued is cache
    assert continued.get_seq_length() == 5
    assert prefix.get_seq_length() == 3
    continued.reset()
    assert continued.get_seq_length() == 0
    replay = hf_model(ids, past_key_values=continued, use_cache=True)
    torch.testing.assert_close(replay.logits, hf_model(ids, use_cache=False).logits, rtol=0, atol=0)
    assert replay.past_key_values.get_seq_length() == 5
    assert continued.get_seq_length() == 5
    assert replay.past_key_values is continued


@pytest.mark.parametrize("decoder_only", [False, True])
@torch.inference_mode()
def test_empty_native_cache_publishes_through_public_output(decoder_only):
    hf_model = model()
    consumer = hf_model.model if decoder_only else hf_model
    cache = DynamicCache()
    ids = torch.tensor([[1, 4]])
    output = consumer(ids, past_key_values=cache, use_cache=True)
    assert len(cache.layers) == hf_model.config.num_hidden_layers
    assert cache.get_seq_length() == 2
    assert output.past_key_values is cache
    assert type(output.past_key_values) is DynamicCache
    assert output.past_key_values.get_seq_length() == 2
    expected = consumer(ids, use_cache=False)
    torch.testing.assert_close(output[0], expected[0], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["static", "subclass", "offloaded", "layer"])
@torch.inference_mode()
def test_unsupported_cache_fails_before_decoder_mutation(kind):
    hf_model = model()
    class UnknownCache(DynamicCache):
        pass
    cache = DynamicCache(config=hf_model.config)
    if kind == "static":
        cache = StaticCache(config=hf_model.config, max_cache_len=16)
    elif kind == "subclass":
        cache = UnknownCache(config=hf_model.config)
    elif kind == "offloaded":
        cache.offloading = True
    else:
        class UnknownLayer(DynamicLayer):
            pass
        cache.layers[0] = UnknownLayer()
    with pytest.raises(TypeError, match="DeepSeek V2.*cache"):
        hf_model(torch.tensor([[1, 4]]), past_key_values=cache, use_cache=True)
    assert all(layer.get_seq_length() == 0 for layer in cache.layers)


def test_static_cache_compilation_is_not_advertised():
    hf_model = model()
    assert hf_model._can_compile_fullgraph is False
    assert hf_model.model._can_compile_fullgraph is False


@pytest.mark.parametrize("kind", ["partial", "unequal", "extra"])
@torch.inference_mode()
def test_invalid_cache_prefix_rejected_without_mutation(kind):
    hf_model = model()
    cache = hf_model(torch.tensor([[1, 4]]), use_cache=True).past_key_values
    if kind == "partial":
        cache.layers.pop()
    elif kind == "unequal":
        cache.layers[0].crop(-1)
    else:
        cache.layers.append(DynamicLayer())
    layers = cache.layers
    states = [(layer, vars(layer).copy()) for layer in layers]
    with pytest.raises(ValueError, match="complete, uniform decoder prefix"):
        hf_model(torch.tensor([[7]]), past_key_values=cache, use_cache=True)
    assert cache.layers is layers
    assert len(cache.layers) == len(states)
    for layer, (original, state) in zip(cache.layers, states):
        assert layer is original
        assert vars(layer).keys() == state.keys()
        assert all(vars(layer)[key] is value for key, value in state.items())


@pytest.mark.parametrize("q_rank,dtype,construction,rope_type", [
    (None, torch.float16, "cast", "default"),
    (8, torch.bfloat16, "from_config", "dynamic"),
    (None, torch.bfloat16, "cast", "yarn"),
    (8, torch.float16, "from_config", "yarn"),
    (None, torch.float16, "from_config", "dynamic"),
])
@torch.inference_mode()
def test_rope_precision_cast_reload_and_generation(tmp_path, q_rank, dtype, construction, rope_type):
    rope = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type != "default":
        rope.update(factor=3.0)
    if rope_type == "yarn":
        rope.update(original_max_position_embeddings=64)
    # Nontrivial attention scores: tiny default 0.02 weights hide rotary errors
    # below BF16 logit rounding despite materially different frequency tables.
    cfg = config(
        q_lora_rank=q_rank, num_hidden_layers=3, qk_rope_head_dim=8,
        rope_parameters=rope, initializer_range=0.2,
    )
    torch.manual_seed(271)
    if construction == "cast":
        hf_model = DeepseekV2ForCausalLM(cfg).eval()
        hf_model.half() if dtype == torch.float16 else hf_model.bfloat16()
    else:
        hf_model = DeepseekV2ForCausalLM._from_config(cfg, dtype=dtype).eval()
    ids = torch.tensor([[1, 4, 7, 9, 12, 13]])
    positions = torch.tensor([[0, 11, 127, 1024, 10000, 30000]])
    expected = hf_model(ids, position_ids=positions, use_cache=False).logits
    # Native computation with the same learned tensors and an uncast FP32 RoPE
    # recipe independently qualifies long-position low-precision LM numerics.
    reference = NativeModel(cfg).eval().to(dtype=dtype)
    reference.load_state_dict(hf_model.state_dict())
    reference.model.rotary_emb = NativeRotaryEmbedding(config=cfg)
    torch.testing.assert_close(expected, reference(ids, position_ids=positions, use_cache=False).logits, rtol=0, atol=0)
    hf_model.save_pretrained(tmp_path, safe_serialization=True)
    restored = DeepseekV2ForCausalLM.from_pretrained(tmp_path, dtype=dtype).eval()
    actual = restored(ids, position_ids=positions, use_cache=False).logits
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in hf_model.state_dict().items())
    generated = hf_model.generate(ids, position_ids=positions, max_new_tokens=2, do_sample=False)
    assert torch.equal(generated, restored.generate(ids, position_ids=positions, max_new_tokens=2, do_sample=False))
    assert torch.equal(generated, reference.generate(ids, position_ids=positions, max_new_tokens=2, do_sample=False))

    # Dynamic growth has already occurred: a cast must retain its FP32 recipe,
    # and short-position replay must reset to the original recipe like native.
    hf_model.float()
    for key, value in hf_model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key].float(), rtol=0, atol=0)
    reference.float()
    for pos in (positions, torch.arange(6)[None]):
        actual = hf_model(ids, position_ids=pos, use_cache=False).logits
        torch.testing.assert_close(actual, reference(ids, position_ids=pos, use_cache=False).logits, rtol=0, atol=0)
    rotary = hf_model.model.rotary_emb
    native = NativeRotaryEmbedding(config=cfg)
    x = torch.zeros(1, 6, cfg.hidden_size)
    if rope_type == "dynamic":
        native(x, torch.as_tensor(rotary.max_seq_len_cached - 1).reshape(1, 1))
    for pos in (positions, torch.arange(6)[None]):
        torch.testing.assert_close(rotary(x, pos), native(x, pos), rtol=0, atol=0)
        assert rotary.inv_freq.dtype == torch.float32
        assert rotary.original_inv_freq.dtype == torch.float32


@pytest.mark.parametrize("routed,shared,freq,expected_moe", [
    (None, None, 1, []), (4, None, 2, [2, 4]), (4, 2, 2, [2, 4]), (4, 2, 1, [1, 2, 3, 4]),
])
@torch.inference_mode()
def test_dense_periodic_and_optional_shared_layout_parity(routed, shared, freq, expected_moe):
    actual = model(num_hidden_layers=5, n_routed_experts=routed, n_shared_experts=shared, moe_layer_freq=freq)
    assert [i for i, layer in enumerate(actual.model.layers) if not isinstance(layer.mlp, DeepseekV2MLP)] == expected_moe
    # Assemble the independently specified layout from native dense/MLA blocks.
    native = NativeModel(tiny_config(num_hidden_layers=5, first_k_dense_replace=5)).eval()
    for i in expected_moe:
        source = NativeModel(tiny_config(num_hidden_layers=1, first_k_dense_replace=0)).eval().model.layers[0].mlp
        if shared is None:
            # No-shared variant's oracle sums actual per-expert native MLPs below.
            source.shared_experts = torch.nn.Identity()
        native.model.layers[i].mlp = source
    # For the no-shared case use zero shared contribution, leaving the routed
    # calculation and complete decoder output as the oracle.
    handles = []
    if shared is None:
        for i in expected_moe:
            handles.append(native.model.layers[i].mlp.shared_experts.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out)))
    native.load_state_dict(actual.state_dict(), strict=True)
    ids = torch.tensor([[1, 9, 8, 6], [1, 3, 4, 7]])
    try:
        torch.testing.assert_close(actual(ids).logits, native(ids).logits, rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
    _convert_model_with_defuser(DeepSeekV2QModel, actual, cleanup_original=True)
    wrapper = DeepSeekV2QModel(actual, quantized=False, quantize_config=QuantizeConfig())
    simple = {
        name.split(":", 1)[0]
        for subset in wrapper.simple_layer_modules(actual.config, wrapper.quantize_config)
        for name in subset
    }
    for layer in actual.model.layers:
        for key in layer.state_dict():
            assert wrapper.module_plan().find(key.rsplit(".", 1)[0]) is not None
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                assert name in simple


@pytest.mark.parametrize("q_rank", [None, 8])
@pytest.mark.parametrize("defused", [False, True], ids=["fused", "checkpoint-names"])
@torch.inference_mode()
def test_safetensors_save_reload_and_expert_conversion(tmp_path, q_rank, defused):
    original = model(q_lora_rank=q_rank, norm_topk_prob=True, moe_layer_freq=2, num_hidden_layers=3)
    if defused:
        _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    ids = torch.tensor([[1, 8, 6, 4]])
    expected = original(ids).logits
    path = tmp_path / "first"
    save_checkpoint(path, original, malicious=True)
    assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
    loaded, info = DeepSeekV2QModel.loader.from_pretrained(path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    loaded.eval()
    _convert_model_with_defuser(DeepSeekV2QModel, loaded, cleanup_original=True)
    _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(loaded(ids).logits, expected, rtol=2e-5, atol=2e-6)
    save_checkpoint(tmp_path / "second", loaded)
    again = DeepSeekV2QModel.loader.from_pretrained(tmp_path / "second").eval()
    torch.testing.assert_close(again(ids).logits, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.inference_mode()
def test_public_load_owns_config_and_model_despite_auto_map(tmp_path, monkeypatch, trust, bypass):
    original = model()
    path = tmp_path / "model"
    save_checkpoint(path, original, malicious=True)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is the tested input.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert resolve_trust_remote_code(path, trust_remote_code=trust) is False
    selected = load_model_config(path, trust_remote_code=trust)
    assert type(selected) is DeepseekV2Config
    loaded = GPTQModel.load(str(path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is DeepSeekV2QModel
    assert type(loaded.model) is DeepseekV2ForCausalLM
    ids = torch.tensor([[1, 6, 3, 7]])
    torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits, rtol=2e-5, atol=2e-6)


@torch.inference_mode()
def test_failed_checkpoint_load_then_corrected_retry(tmp_path):
    original = model()
    path = tmp_path / "retry"
    save_checkpoint(path, original)
    weights = load_file(path / "model.safetensors")
    good = weights["model.layers.0.self_attn.q_b_proj.weight"].clone()
    weights["model.layers.0.self_attn.q_b_proj.weight"] = good[:1]
    save_file(weights, path / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match="(?i)(size|shape|mismatch)"):
        DeepSeekV2QModel.loader.from_pretrained(path)
    weights["model.layers.0.self_attn.q_b_proj.weight"] = good
    save_file(weights, path / "model.safetensors")
    retry = DeepSeekV2QModel.loader.from_pretrained(path).eval()
    ids = torch.tensor([[1, 3, 9]])
    torch.testing.assert_close(retry(ids).logits, original(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("defused", [False, True], ids=["fused", "checkpoint-names"])
@pytest.mark.parametrize("routed,shared,freq", [(4, 2, 1), (None, None, 1), (4, None, 2), (4, 2, 2)])
@torch.inference_mode()
def test_public_quantization_shell_materializes_exact_checkpoint_weights(tmp_path, defused, routed, shared, freq):
    original = model(num_hidden_layers=3, n_routed_experts=routed, n_shared_experts=shared, moe_layer_freq=freq)
    if defused:
        _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    path = tmp_path / "lazy"
    save_checkpoint(path, original, malicious=True)
    wrapped = GPTQModel.load(
        str(path), quantize_config=QuantizeConfig(device="cpu", group_size=-1, offload_to_disk=True),
        dtype=torch.float32,
    )
    assert wrapped.turtle_model is not None
    assert DeepSeekV2QModel.resolve_hf_conversion_map_reversed(wrapped.model) is not None
    wrapped.shell_module_materialize(wrapped.model, torch.device("cpu"), module_path="")
    _convert_model_with_defuser(DeepSeekV2QModel, original, cleanup_original=True)
    assert set(wrapped.model.state_dict()) == set(original.state_dict())
    for key, value in original.state_dict().items():
        actual = wrapped.model.state_dict()[key]
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    ids = torch.tensor([[1, 8, 5, 7]])
    torch.testing.assert_close(wrapped.model(ids).logits, original(ids).logits, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("q_rank", [None, 8], ids=["lite-yarn", "full-yarn"])
@torch.inference_mode()
def test_legacy_yarn_metadata_and_native_cache_roundtrip(tmp_path, q_rank):
    # Same YaRN fields as pinned DeepSeek-Coder-V2-Lite-Instruct, with tiny
    # original context so the test exercises positions beyond that context.
    rope = {
        "type": "yarn", "factor": 40.0, "beta_fast": 32.0, "beta_slow": 1.0,
        "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 4,
    }
    hf_model = model(q_lora_rank=q_rank, max_position_embeddings=160, rope_scaling=rope, rope_theta=10000.0)
    save_checkpoint(tmp_path, hf_model, malicious=True)
    selected = load_model_config(tmp_path, trust_remote_code=True)
    assert selected.rope_parameters["rope_type"] == "yarn"
    assert selected.rope_parameters["factor"] == 40.0
    assert selected.rope_parameters["mscale_all_dim"] == 0.707
    loaded = DeepSeekV2QModel.loader.from_pretrained(tmp_path).eval()
    ids = torch.tensor([[1, 9, 5, 6, 7, 8]])
    expected = hf_model(ids, use_cache=False).logits
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)
    prefix = loaded(ids[:, :4], use_cache=True)
    suffix = loaded(ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(suffix.logits, expected[:, 4:], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("overrides,match", [
    ({"moe_layer_freq": 0}, "moe_layer_freq"), ({"ep_size": 2}, "ep_size"),
    ({"scoring_func": "sigmoid"}, "scoring_func"), ({"topk_method": "noaux_tc"}, "topk_method"),
    ({"n_routed_experts": 3, "topk_method": "group_limited_greedy"}, "group-limited"),
    ({"num_experts_per_tok": 8}, "num_experts_per_tok"),
])
def test_ambiguous_or_unsupported_configuration_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


@pytest.mark.parametrize("layer_index", [0, 1])
@torch.inference_mode()
def test_actual_awq_plain_quantization_reaches_all_decoder_linears(layer_index):
    hf_model = model()
    _convert_model_with_defuser(DeepSeekV2QModel, hf_model, cleanup_original=True)
    wrapper = DeepSeekV2QModel(hf_model, quantized=False, quantize_config=QuantizeConfig())
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=-1),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=hf_model)
    processor.draw_progress = lambda *a, **kw: None
    processor.log_new_row = lambda *a, **kw: None
    state = processor._get_layer_state(layer_index)
    state.layer_module = hf_model.model.layers[layer_index]
    linears = {
        name: NamedModule(module, name, f"model.layers.{layer_index}.{name}", layer_index)
        for name, module in state.layer_module.named_modules() if isinstance(module, torch.nn.Linear)
    }
    state.modules.update(linears)
    before = {name: item.weight.clone() for name, item in linears.items()}
    dense = {key: value.clone() for key, value in state.layer_module.state_dict().items() if key.rsplit(".", 1)[0] not in linears}
    processor._quantize_layer(layer_index, state)
    assert state.quantized and not state.modules
    for name, item in linears.items():
        assert "q_scales" in item.state and "q_zeros" in item.state
        assert not torch.equal(item.weight, before[name]), name
    for key, value in dense.items():
        torch.testing.assert_close(hf_model.model.layers[layer_index].state_dict()[key], value, rtol=0, atol=0)
    assert torch.isfinite(hf_model(torch.tensor([[1, 3, 6]])).logits).all()
