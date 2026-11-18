# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real converted DBRX qualification; no checkpoint Python or trained weights."""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.dbrx_converted import DbrxConvertedQModel
from gptqmodel.utils.hf import load_model_config
from gptqmodel.utils.security import BYPASS_ENV


def tiny_payload(**overrides):
    payload = {
        "model_type": "dbrx_converted", "d_model": 32, "n_heads": 4, "n_layers": 2, "max_seq_len": 64,
        "vocab_size": 41, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": None,
        "attn_config": {"kv_n_heads": 2, "clip_qkv": 0.15, "rope_theta": 500000},
        "ffn_config": {"ffn_hidden_size": 48, "moe_num_experts": 3, "moe_top_k": 2},
    }
    payload.update(overrides)
    return payload


def make_model(attn_implementation=None, **overrides):
    from gptqmodel.models.ports.dbrx_converted.configuration_dbrx import DbrxConfig
    from gptqmodel.models.ports.dbrx_converted.modeling_dbrx import DbrxForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(143)
        config = DbrxConfig(**tiny_payload(**overrides), attn_implementation=attn_implementation)
        model = DbrxForCausalLM(config).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.ndim == 1:
                    parameter.copy_(torch.linspace(0.6, 1.4, parameter.numel()))
                else:
                    parameter.normal_(0, 0.16)
    return model


def poison_checkpoint(path, payload):
    path.mkdir(exist_ok=True)
    marker = path / "EXECUTED"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    (path / "evil_config.py").write_text(code)
    (path / "evil_model.py").write_text(code)
    payload["auto_map"] = {"AutoConfig": "evil_config.EvilConfig", "AutoModelForCausalLM": "evil_model.EvilModel"}
    (path / "config.json").write_text(json.dumps(payload))
    return marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_definition_owns_config_before_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is the contract under test.
    marker = poison_checkpoint(tmp_path, tiny_payload())
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__ == "gptqmodel.models.ports.dbrx_converted.configuration_dbrx"
    assert config.model_type == "dbrx_converted"
    assert not DbrxConvertedQModel.require_trust_remote_code
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_loader_ignores_auto_map_and_roundtrips_safetensors(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    model = make_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    payload = json.loads((tmp_path / "config.json").read_text())
    marker = poison_checkpoint(tmp_path, payload)
    loader = DbrxConvertedQModel.loader
    config = load_model_config(tmp_path, trust_remote_code=trust)
    with torch.device("meta"):
        shell = loader.from_config(config, trust_remote_code=trust, code_revision="evil")
    assert all(t.device.type == "meta" for t in shell.state_dict().values())
    restored, info = loader.from_pretrained(
        tmp_path, trust_remote_code=trust, code_revision="evil", output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert type(restored) is type(model)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin")) and not list(tmp_path.glob("*.pt"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    ids = torch.tensor([[1, 3, 7, 11], [1, 5, 9, 13]])
    with torch.inference_mode():
        expected = model(ids).logits
        actual = restored(ids).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(model.generate(ids, max_new_tokens=3), restored.generate(ids, max_new_tokens=3))
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("for_quantization", [False, True])
def test_public_gptqmodel_load_uses_local_architecture(tmp_path, monkeypatch, trust, bypass, for_quantization):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    model = make_model(max_seq_len=8192)
    model.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(41)}, unk_token="0")),
        pad_token="0", bos_token="1",
    )
    tokenizer.save_pretrained(tmp_path)
    marker = poison_checkpoint(tmp_path, json.loads((tmp_path / "config.json").read_text()))
    quantize_config = QuantizeConfig(bits=4, group_size=16, offload_to_disk=False) if for_quantization else None
    wrapped = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
        quantize_config=quantize_config,
    )
    assert isinstance(wrapped, DbrxConvertedQModel)
    assert type(wrapped.model) is type(model)
    assert wrapped.model.config.max_seq_len == 8192
    if for_quantization:
        assert wrapped.model.seqlen == 8192
    assert wrapped.module_plan().find("ffn.experts.mlp.2.w2").quantize
    ids = torch.tensor([[1, 3, 7, 11]])
    with torch.inference_mode():
        torch.testing.assert_close(wrapped.model(ids).logits, model(ids).logits, rtol=0, atol=0)
        assert torch.equal(wrapped.generate(ids, max_new_tokens=3), model.generate(ids, max_new_tokens=3))
    assert not marker.exists()


def reference_logits(model, ids, mask=None, positions=None):
    """Independent dense Torch evaluation of the pinned converted equations.

    Unlike the port, evaluate every expert then select its contribution with a
    dense routing mask; use functional projections and explicit per-head RoPE.
    """
    config, state = model.config, model.state_dict()
    hidden = F.embedding(ids, state["transformer.wte.weight"])
    batch, length = ids.shape
    positions = torch.arange(length)[None] if positions is None else positions
    head_dim = config.d_model // config.n_heads
    for index in range(config.n_layers):
        root = f"transformer.blocks.{index}."
        def weight(name):
            return state[root + name + ".weight"]
        norm = F.layer_norm(hidden, (config.d_model,), weight("norm_attn_norm.norm_1"), eps=1e-5)
        projections = [F.linear(norm, weight("norm_attn_norm.attn." + name)) for name in ("q_proj", "k_proj", "v_proj")]
        if config.attn_config.clip_qkv is not None:
            projections = [value.clamp(-config.attn_config.clip_qkv, config.attn_config.clip_qkv) for value in projections]
        q, k, v = [value.reshape(batch, length, -1, head_dim).transpose(1, 2) for value in projections]
        angle = positions[..., None].float() / config.attn_config.rope_theta ** (torch.arange(0, head_dim, 2) / head_dim)
        cosine, sine = angle.cos()[:, None].to(hidden.dtype), angle.sin()[:, None].to(hidden.dtype)
        def rotary(value):
            left, right = value.chunk(2, dim=-1)
            return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(value.dtype)
        q, k = rotary(q), rotary(k)
        heads = torch.arange(config.n_heads) // (config.n_heads // config.attn_config.kv_n_heads)
        k, v = k[:, heads], v[:, heads]
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * head_dim ** -0.5
        invalid = torch.ones(length, length, dtype=torch.bool).triu(1)[None, None].expand(batch, 1, -1, -1)
        if mask is not None:
            invalid = invalid | ~mask[:, None, None].bool()
        scores = scores.masked_fill(invalid, torch.finfo(scores.dtype).min)
        attention = torch.einsum("bhqk,bhkd->bhqd", scores.float().softmax(-1).to(v.dtype), v)
        hidden = hidden + F.linear(attention.transpose(1, 2).reshape(batch, length, -1), weight("norm_attn_norm.attn.out_proj"))
        norm = F.layer_norm(hidden, (config.d_model,), weight("norm_attn_norm.norm_2"), eps=1e-5)
        probabilities = F.linear(norm.reshape(-1, config.d_model), weight("ffn.router.layer")).float().softmax(-1)
        top_weights, top_experts = probabilities.topk(config.ffn_config.moe_top_k, -1)
        if config.ffn_config.moe_normalize_expert_weights:
            top_weights = top_weights / top_weights.norm(p=config.ffn_config.moe_normalize_expert_weights, dim=-1, keepdim=True)
        if config.ffn_config.uniform_expert_assignment:
            top_experts = (torch.arange(top_experts.numel()) % config.ffn_config.moe_num_experts).reshape_as(top_experts)
        routing = torch.zeros_like(probabilities, dtype=norm.dtype).scatter_add(1, top_experts, top_weights.to(norm.dtype))
        output = torch.zeros_like(norm)
        activation = dict(config.ffn_config.ffn_act_fn)
        function = getattr(F, activation.pop("name"))
        for expert in range(config.ffn_config.moe_num_experts):
            prefix = f"ffn.experts.mlp.{expert}."
            gate = function(F.linear(norm, weight(prefix + "w1")), **activation)
            expert_output = F.linear(gate * F.linear(norm, weight(prefix + "v1")), weight(prefix + "w2"))
            output += expert_output * routing[:, expert].reshape(batch, length, 1)
        hidden = hidden + output
    hidden = F.layer_norm(hidden, (config.d_model,), state["transformer.norm_f.weight"], eps=1e-5)
    return F.linear(hidden, state["lm_head.weight"]).float()


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("ffn", [
    {"ffn_hidden_size": 48, "moe_num_experts": 3, "moe_top_k": 2},
    {"ffn_hidden_size": 48, "moe_num_experts": 4, "moe_top_k": 1, "moe_normalize_expert_weights": None},
    {"ffn_hidden_size": 48, "moe_num_experts": 4, "moe_top_k": 3, "moe_normalize_expert_weights": 2, "uniform_expert_assignment": True,
         "ffn_act_fn": {"name": "gelu", "approximate": "tanh"}},
])
def test_real_forward_matches_independent_converted_equations(kv_heads, ffn):
    model = make_model("eager", attn_config={"kv_n_heads": kv_heads, "clip_qkv": 0.15, "rope_theta": 500000}, ffn_config=ffn)
    ids = torch.tensor([[0, 0, 1, 3, 7], [1, 2, 5, 8, 9]])
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, position_ids=positions, use_cache=False, output_attentions=True,
                       output_router_logits=True, output_hidden_states=True)
        expected = reference_logits(model, ids, mask, positions)
        torch.testing.assert_close(actual.logits, expected, rtol=2e-5, atol=2e-6)
        assert actual.logits.std() > 0.1
        # The attention interface dispatches SDPA to the same equations (fully masked pad rows are unspecified).
        sdpa = make_model("sdpa", attn_config={"kv_n_heads": kv_heads, "clip_qkv": 0.15, "rope_theta": 500000}, ffn_config=ffn)
        assert sdpa.config._attn_implementation == "sdpa"
        sdpa_logits = sdpa(ids, attention_mask=mask, position_ids=positions).logits
        torch.testing.assert_close(sdpa_logits[mask.bool()], expected[mask.bool()], rtol=2e-5, atol=2e-6)
        assert len(actual.attentions) == model.config.n_layers
        assert len(actual.hidden_states) == model.config.n_layers + 1
        assert torch.isfinite(actual.aux_loss)
        for weights in actual.router_logits:
            torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[0]))
        # Clipping and causal masking are exercised, not just configured.
        norm = model.transformer.blocks[0].norm_attn_norm.norm_1(model.transformer.wte(ids))
        assert model.transformer.blocks[0].norm_attn_norm.attn.q_proj(norm).abs().max() > 0.15
        torch.testing.assert_close(actual.attentions[0][..., -2, -1], torch.zeros(2, 4))
        altered = ids.clone()
        altered[:, -1] = 17
        valid = mask[:, :-1].bool()
        torch.testing.assert_close(
            model(altered, attention_mask=mask, position_ids=positions).logits[:, :-1][valid], actual.logits[:, :-1][valid],
        )


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("beams", [1, 3])
def test_cache_continuation_generation_and_invalid_retry(static, beams):
    from transformers.cache_utils import DynamicCache, StaticCache

    model = make_model()
    ids = torch.tensor([[0, 1, 3, 7, 11, 19], [1, 2, 5, 9, 13, 23]])
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    cache = StaticCache(config=model.config, max_cache_len=16) if static else DynamicCache(config=model.config)
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        first = model(ids[:, :3], attention_mask=mask[:, :3], position_ids=positions[:, :3], past_key_values=cache)
        assert first.past_key_values is cache and cache.get_seq_length() == 3
        for start, end in [(3, 5), (5, 6)]:
            continued = model(ids[:, start:end], attention_mask=mask[:, :end], position_ids=positions[:, start:end], past_key_values=cache)
            torch.testing.assert_close(continued.logits, full[:, start:end], rtol=2e-5, atol=2e-6)
            assert cache.get_seq_length() == end
        cached = model.generate(ids, attention_mask=mask, max_new_tokens=4, do_sample=False,
                                num_beams=beams,
                                cache_implementation="static" if static else "dynamic")
        fresh = model.generate(ids, attention_mask=mask, max_new_tokens=4, do_sample=False, use_cache=False, num_beams=beams)
        assert torch.equal(cached, fresh)
        assert cached.shape == (2, 10)
        # A new call must not inherit caller-owned past state.
        torch.testing.assert_close(model(ids, attention_mask=mask, position_ids=positions).logits, full)


def test_loss_inputs_embeds_and_tuple_contract():
    model = make_model()
    ids = torch.tensor([[1, 3, 7, 11, 19]])
    labels = ids.clone()
    labels[:, 1] = -100
    with torch.inference_mode():
        result = model(ids, labels=labels, use_cache=False, output_router_logits=True)
        ce = F.cross_entropy(result.logits[:, :-1].reshape(-1, 41), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(result.loss, ce + model.config.router_aux_loss_coef * result.aux_loss)
        embedded = model(inputs_embeds=model.get_input_embeddings()(ids), use_cache=False)
        torch.testing.assert_close(embedded.logits, result.logits)
        plain = model(ids, use_cache=False, return_dict=False)
        assert len(plain) == 1
        torch.testing.assert_close(plain[0], result.logits)
        torch.testing.assert_close(model(ids, logits_to_keep=2).logits, result.logits[:, -2:])
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))


def test_router_aux_loss_with_padded_cache_continuation():
    model = make_model(output_router_logits=True)
    ids = torch.tensor([[0, 1, 3, 7, 11, 19], [1, 2, 5, 9, 13, 23]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        prefix = model(ids[:, :4], attention_mask=mask[:, :4])
        continued = model(ids[:, 4:], attention_mask=mask, past_key_values=prefix.past_key_values)
        routing = torch.cat(continued.router_logits).softmax(-1)
        experts = routing.topk(model.num_experts_per_tok, dim=-1).indices
        assignment = F.one_hot(experts, model.num_experts).float().mean(0)
        expected = (assignment * routing.mean(0)[None]).sum() * model.num_experts
        torch.testing.assert_close(continued.aux_loss, expected)
        assert continued.past_key_values.get_seq_length() == ids.shape[1]
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3)
        fresh = model.generate(ids, attention_mask=mask, max_new_tokens=3, use_cache=False)
        assert torch.equal(generated, fresh)


@pytest.mark.parametrize("num_beams", [1, 2])
@pytest.mark.parametrize("padded", [False, True])
def test_static_generation_preserves_router_padding_mask(num_beams, padded):
    model = make_model(output_router_logits=True)
    ids = torch.tensor([[1, 3, 7, 11, 19], [1, 2, 5, 9, 13]])
    if padded:
        ids[0, :2] = 0
    mask = (ids != 0).long()
    with torch.inference_mode():
        options = {"max_new_tokens": 3, "num_beams": num_beams, "return_dict_in_generate": True,
                   "output_scores": True}
        seen = []

        def capture(module, args, kwargs, output):
            seen.append((kwargs.get("attention_mask"), output.aux_loss))

        handle = model.register_forward_hook(capture, with_kwargs=True)
        try:
            static = model.generate(ids, attention_mask=mask, cache_implementation="static", **options)
        finally:
            handle.remove()
        assert len(seen) == 3
        assert all(torch.isfinite(aux_loss) for _, aux_loss in seen)
        assert all(padding_mask is None or padding_mask.ndim == 2 for padding_mask, _ in seen)
        if padded:
            assert all(padding_mask is not None for padding_mask, _ in seen)
        dynamic = model.generate(ids, attention_mask=mask, cache_implementation="dynamic", **options)
        fresh = model.generate(ids, attention_mask=mask, use_cache=False, **options)
        assert torch.equal(static.sequences, dynamic.sequences)
        assert torch.equal(static.sequences, fresh.sequences)
        for actual, expected in zip(static.scores, fresh.scores):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("padded", [False, True])
def test_static_capacity_mask_uses_current_physical_positions(padded):
    from transformers import StaticCache

    model = make_model(output_router_logits=True)
    ids = torch.tensor([[1, 3, 7, 11, 19], [1, 2, 5, 9, 13]])
    if padded:
        ids[0, :2] = 0
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    cache = StaticCache(config=model.config, max_cache_len=16)
    control_cache = StaticCache(config=model.config, max_cache_len=16)
    prefix_mask = F.pad(mask[:, :3], (0, 13))
    full_mask = F.pad(mask, (0, 11))
    with torch.inference_mode():
        prefix = model(ids[:, :3], attention_mask=prefix_mask, position_ids=positions[:, :3], past_key_values=cache)
        model(ids[:, :3], attention_mask=mask[:, :3], position_ids=positions[:, :3], past_key_values=control_cache)
        continued = model(ids[:, 3:], attention_mask=full_mask, position_ids=positions[:, 3:], past_key_values=cache,
                          labels=ids[:, 3:])
        control = model(ids[:, 3:], attention_mask=mask, position_ids=positions[:, 3:], past_key_values=control_cache,
                        labels=ids[:, 3:])
        assert torch.isfinite(prefix.aux_loss)
        assert torch.isfinite(continued.aux_loss)
        assert torch.isfinite(continued.loss)
        torch.testing.assert_close(continued.logits, control.logits, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(continued.aux_loss, control.aux_loss, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(continued.loss, control.loss, atol=3e-6, rtol=3e-6)
        routing = torch.cat(continued.router_logits).softmax(-1)
        assignment = F.one_hot(routing.topk(model.num_experts_per_tok, dim=-1).indices, model.num_experts).float()
        expected_aux = (assignment.mean(0) * routing.mean(0)[None]).sum() * model.num_experts
        torch.testing.assert_close(continued.aux_loss, expected_aux)


def test_arbitrary_4d_mask_is_not_router_padding_metadata():
    from transformers import DynamicCache

    model = make_model(output_router_logits=True)
    cache = DynamicCache(config=model.config)
    ids = torch.tensor([[1, 3, 7]])
    mask = torch.zeros(1, 1, 3, 3)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="2D padding mask"):
            model(ids, attention_mask=mask, past_key_values=cache)
        assert [int(layer.get_seq_length()) for layer in cache.layers] == [0, 0]
        actual = model(ids, attention_mask=torch.ones_like(ids), past_key_values=cache)
        control = model(ids, use_cache=False)
        torch.testing.assert_close(actual.logits, control.logits)


def test_4d_additive_attention_mask_still_works_without_router_loss():
    model = make_model("eager")
    ids = torch.tensor([[1, 3, 7]])
    mask = torch.full((1, 1, 3, 3), torch.finfo(torch.float32).min).triu(1)
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, use_cache=False, output_attentions=True)
        control = model(ids, use_cache=False, output_attentions=True)
        torch.testing.assert_close(actual.logits, control.logits)
        for attention, expected in zip(actual.attentions, control.attentions):
            torch.testing.assert_close(attention, expected)


def test_parallel_forward_uses_independent_caller_owned_caches():
    model = make_model()
    inputs = [torch.tensor([[1, 3, 7, 11]]), torch.tensor([[2, 5, 9, 13]])]
    with torch.inference_mode():
        expected = [model(ids, use_cache=False).logits for ids in inputs]
    barrier = Barrier(2)
    def run(ids):
        with torch.inference_mode():
            barrier.wait(timeout=10)
            first = model(ids[:, :2])
            last = model(ids[:, 2:], past_key_values=first.past_key_values)
            return torch.cat((first.logits, last.logits), 1), last.past_key_values
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(run, inputs))
    assert actual[0][1] is not actual[1][1]
    for index, (logits, cache) in enumerate(actual):
        torch.testing.assert_close(logits, expected[index], rtol=2e-5, atol=2e-6)
        assert cache.get_seq_length() == 4


@pytest.mark.parametrize("overrides,match", [
    ({"tie_word_embeddings": True}, "tie_word_embeddings"),
    ({"d_model": 30}, "divisible GQA"),
    ({"ffn_config": {"moe_num_experts": 2, "moe_top_k": 3}}, "moe_top_k"),
])
def test_invalid_config_fails_before_construction(overrides, match):
    from gptqmodel.models.ports.dbrx_converted.configuration_dbrx import DbrxConfig

    with pytest.raises(ValueError, match=match):
        DbrxConfig(**tiny_payload(**overrides))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_forward_and_safetensors_reload(tmp_path, dtype):
    model = make_model().to(dtype)
    ids = torch.tensor([[1, 3, 7, 11], [2, 5, 9, 13]])
    with torch.inference_mode():
        result = model(ids, output_hidden_states=True)
        assert result.logits.dtype == torch.float32  # Upstream LM-head contract.
        assert result.hidden_states[-1].dtype == dtype
        assert torch.isfinite(result.logits).all()
        reference = reference_logits(model, ids)
        torch.testing.assert_close(result.logits, reference, rtol=0.02, atol=0.02)
        model.save_pretrained(tmp_path, safe_serialization=True)
        loaded = DbrxConvertedQModel.loader.from_pretrained(tmp_path, dtype=dtype)
        torch.testing.assert_close(loaded(ids).logits, result.logits, rtol=0, atol=0)


def test_plain_awq_public_quantization_and_packed_reload(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = make_model(ffn_config={"ffn_hidden_size": 48, "moe_num_experts": 3, "moe_top_k": 3})
    model.save_pretrained(tmp_path, safe_serialization=True)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(41)}, unk_token="0")),
        pad_token="0", bos_token="1",
    ).save_pretrained(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, offload_to_disk=False, fallback=None)
    wrapped = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[1, 3, 7, 11, 19, 23, 29, 37]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapped.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name: module for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 26
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    for layer in wrapped.model.transformer.blocks:
        assert isinstance(layer.ffn.router.layer, torch.nn.Linear)
        assert isinstance(layer.norm_attn_norm.norm_1, torch.nn.LayerNorm)
    with torch.inference_mode():
        expected = wrapped.model(**calibration[0], use_cache=False).logits
        generated = wrapped.generate(**calibration[0], max_new_tokens=3, do_sample=False)
    saved = tmp_path / "quantized"
    wrapped.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors"))
    assert not list(saved.glob("*.bin")) and not list(saved.glob("*.pt"))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in loaded.model.modules()) == 26
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
        assert torch.equal(loaded.generate(**calibration[0], max_new_tokens=3, do_sample=False), generated)
