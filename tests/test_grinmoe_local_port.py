# SPDX-License-Identifier: Apache-2.0
"""Real GRIN-MoE runtime tests; no checkpoint Python or pretrained weights needed."""

import json

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.grinmoe import GrinMoeQModel
from gptqmodel.models.ports.grinmoe.configuration_grinmoe import GRINMoEConfig
from gptqmodel.models.ports.grinmoe.modeling_grinmoe import GRINMoEForCausalLM, sparsemixer
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(**kwargs):
    values = {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 3,
              "max_position_embeddings": 32, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
              "attention_bias": True, "lm_head_bias": True, "router_jitter_noise": 0.2}
    values.update(kwargs)
    return GRINMoEConfig(**values)


def tiny_model(**kwargs):
    torch.manual_seed(103)
    return GRINMoEForCausalLM(tiny_config(**kwargs)).eval()


def poisoned_checkpoint(path, **kwargs):
    model = tiny_model(**kwargs)
    model.save_pretrained(path)
    marker = path / "executed"
    (path / "checkpoint_code.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('EXECUTED')\nraise RuntimeError('checkpoint code')\n"
    )
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "checkpoint_code.Config",
                           "AutoModelForCausalLM": "checkpoint_code.Model"}
    payload["architectures"] = ["GRIN-MoE"]  # Exact released checkpoint spelling.
    (path / "config.json").write_text(json.dumps(payload))
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
                           eos_token="[EOS]", unk_token="[UNK]").save_pretrained(path)
    return model, marker


@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_definition_owned_selection_ignores_checkpoint_code(tmp_path, monkeypatch, trust, bypass):
    model, marker = poisoned_checkpoint(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security permission, not model behavior selection.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    failure = None
    try:
        config = load_model_config(tmp_path, trust_remote_code=trust)
    except Exception as error:
        failure = error
    assert failure is None, f"A supported GRIN-MoE must select package config, not checkpoint code: {failure!r}"
    assert type(config) is GRINMoEConfig
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is GrinMoeQModel
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    loaded = GrinMoeQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust, code_revision="ignored")
    assert type(loaded) is GRINMoEForCausalLM
    assert not marker.exists()
    ids = torch.tensor([[1, 4, 9, 7]])
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("trust,bypass", [(False, False), (True, True)])
def test_public_dense_load_and_generation(tmp_path, monkeypatch, trust, bypass):
    model, marker = poisoned_checkpoint(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(wrapper) is GrinMoeQModel
    assert type(wrapper.model) is GRINMoEForCausalLM
    assert wrapper.module_plan().find("block_sparse_moe.experts.2.w1").quantize
    ids = torch.tensor([[1, 5, 7]])
    torch.testing.assert_close(wrapper.model(ids).logits, model(ids).logits, atol=0, rtol=0)
    assert torch.equal(wrapper.model.generate(ids, max_new_tokens=3, do_sample=False),
                       model.generate(ids, max_new_tokens=3, do_sample=False))
    assert not marker.exists()


def reference_routing(scores, epsilon):
    """Scalar/token oracle for upstream two separate masked softmax distributions."""
    weights, indices = [], []
    for row in scores:
        selected, multipliers = [], []
        for _ in range(2):
            remaining = [i for i in range(row.numel()) if i not in selected]
            best = max(remaining, key=lambda i: row[i].item())
            maximum = row[best]
            accepted = [i for i in remaining if not bool(
                (maximum - row[i]) / torch.maximum(row[i].abs(), maximum) > 2 * epsilon
            )]
            probs = F.softmax(row[accepted], dim=0)
            multipliers.append(probs[accepted.index(best)])
            selected.append(best)
        indices.append(selected)
        weights.append(torch.stack(multipliers))
    return torch.stack(weights), torch.tensor(indices, device=scores.device)


def reference_forward(model, ids, mask, positions, *, window=None):
    """Functional source-equation oracle; does not call port modules or cache helpers."""
    state, config = model.state_dict(), model.config
    hidden = F.embedding(ids, state["model.embed_tokens.weight"])
    states, routers = [hidden], []
    batch, seq, width = hidden.shape
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = width // heads
    inv = 1 / config.rope_theta ** (torch.arange(0, dim, 2).float() / dim)
    angles = torch.einsum("bs,d->bsd", positions.float(), inv)
    cos = torch.cat([angles, angles], -1).cos()[:, None].to(hidden.dtype)
    sin = torch.cat([angles, angles], -1).sin()[:, None].to(hidden.dtype)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."

        def linear(x, name):
            return F.linear(x, state[prefix + name + ".weight"], state.get(prefix + name + ".bias"))

        def norm(x, name):
            return F.layer_norm(x, (width,), state[prefix + name + ".weight"],
                                state[prefix + name + ".bias"], config.rms_norm_eps)

        x = norm(hidden, "input_layernorm")
        q = linear(x, "self_attn.q_proj").reshape(batch, seq, heads, dim).transpose(1, 2)
        k = linear(x, "self_attn.k_proj").reshape(batch, seq, kv_heads, dim).transpose(1, 2)
        v = linear(x, "self_attn.v_proj").reshape(batch, seq, kv_heads, dim).transpose(1, 2)
        q = q * cos + torch.cat([-q[..., dim // 2:], q[..., :dim // 2]], -1) * sin
        k = k * cos + torch.cat([-k[..., dim // 2:], k[..., :dim // 2]], -1) * sin
        k = k.repeat_interleave(heads // kv_heads, 1)
        v = v.repeat_interleave(heads // kv_heads, 1)
        scores = (q @ k.transpose(-1, -2)) / dim ** 0.5
        valid = torch.ones(seq, seq, dtype=torch.bool).tril()
        if window is not None:
            valid &= torch.ones(seq, seq, dtype=torch.bool).triu(-window)
        valid = valid[None, None] & mask[:, None, None].bool()
        scores = scores + torch.zeros_like(scores).masked_fill(~valid, torch.finfo(hidden.dtype).min)
        probabilities = scores.float().softmax(-1).to(hidden.dtype)
        attention = (probabilities @ v).transpose(1, 2).reshape(batch, seq, width)
        hidden = hidden + linear(attention, "self_attn.o_proj")
        x = norm(hidden, "post_attention_layernorm").reshape(-1, width)
        router = linear(x, "block_sparse_moe.gate")
        weights, experts = reference_routing(router, config.router_jitter_noise)
        result = torch.zeros_like(x)
        for token in range(x.shape[0]):
            for slot in range(2):
                name = f"block_sparse_moe.experts.{experts[token, slot].item()}."
                expert = linear(F.silu(linear(x[token], name + "w1")) * linear(x[token], name + "w3"), name + "w2")
                result[token] += expert * weights[token, slot]
        hidden = hidden + result.reshape(batch, seq, width)
        routers.append(router)
        states.append(hidden)
    hidden = F.layer_norm(hidden, (width,), state["model.norm.weight"], state["model.norm.bias"], config.rms_norm_eps)
    states[-1] = hidden
    logits = F.linear(hidden, state["lm_head.weight"], state.get("lm_head.bias")).float()
    return logits, states, routers


def test_sparsemixer_is_not_normalized_top2_softmax():
    scores = torch.tensor([[4.0, 3.9, -1.0], [-2.0, -2.01, -8.0], [0.0, 0.0, -0.3], [9.0, 1.0, -5.0]])
    weights, indices = sparsemixer(scores, 2, 0.1, False)
    expected, selected = reference_routing(scores, 0.1)
    torch.testing.assert_close(weights, expected)
    assert torch.equal(indices, selected)
    assert not torch.allclose(weights.sum(-1), torch.ones(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_retains_upstream_fp32_phases_after_low_precision_cast(dtype):
    model = tiny_model(hidden_size=64, max_position_embeddings=8192, rope_theta=10000.0).to(dtype)
    rotary = model.model.layers[0].self_attn.rotary_emb
    positions = torch.tensor([[0, 127, 4095], [7, 511, 8191]])
    dim = model.config.hidden_size // model.config.num_attention_heads
    # Upstream builds its cosine/sine tables from FP32 frequencies before casting.
    inverse = 1 / 10000.0 ** (torch.arange(0, dim, 2).float() / dim)
    phases = positions[..., None] * inverse
    phases = torch.cat([phases, phases], -1)[:, None]
    cos, sin = rotary(torch.zeros(1, dtype=dtype), positions)
    torch.testing.assert_close(cos, phases.cos().to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(sin, phases.sin().to(dtype), atol=0, rtol=0)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("bias,kv_heads", [(True, 1), (False, 2), (True, 4)])
def test_full_model_matches_source_equations(implementation, bias, kv_heads):
    model = tiny_model(attn_implementation=implementation, attention_bias=bias, lm_head_bias=bias,
                       num_key_value_heads=kv_heads, sliding_window=3)
    ids = torch.tensor([[1, 4, 8, 2, 5], [0, 1, 3, 7, 9]])
    mask = ids.ne(0).long()
    positions = torch.tensor([[3, 4, 5, 6, 7], [0, 2, 3, 4, 5]])
    output = model(ids, attention_mask=mask, position_ids=positions, use_cache=False,
                   output_hidden_states=True, output_router_logits=True, labels=ids)
    expected, states, routers = reference_forward(model, ids, mask, positions,
                                                 window=3 if implementation == "eager" else None)
    torch.testing.assert_close(output.logits, expected, atol=1e-7, rtol=1e-5)
    for actual, wanted in zip(output.hidden_states, states):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-5)
    for actual, wanted in zip(output.router_logits, routers):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-5)
    ce = F.cross_entropy(expected[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(output.loss, ce + model.config.router_aux_loss_coef * output.aux_loss)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_cache_continuation_generation_and_causality(implementation, dtype):
    model = tiny_model(attn_implementation=implementation, sliding_window=3).to(dtype)
    ids = torch.tensor([[1, 4, 8, 2, 5, 6], [0, 1, 3, 7, 9, 4]])
    mask = ids.ne(0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    full = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    prefill = model(ids[:, :4], attention_mask=mask[:, :4], position_ids=positions[:, :4])
    assert isinstance(prefill.past_key_values, DynamicCache)
    cache = prefill.past_key_values
    continuation = model(ids[:, 4:], attention_mask=mask, position_ids=positions[:, 4:], past_key_values=cache)
    assert continuation.past_key_values is cache
    assert cache.get_seq_length() == 6
    torch.testing.assert_close(continuation.logits, full[:, 4:], atol=2e-3 if dtype == torch.bfloat16 else 1e-7,
                               rtol=2e-2 if dtype == torch.bfloat16 else 1e-5)
    changed = ids.clone()
    changed[:, -1] = 15
    active = mask[:, :-1].bool()
    torch.testing.assert_close(model(changed, attention_mask=mask, position_ids=positions,
                                    use_cache=False).logits[:, :-1][active], full[:, :-1][active],
                               atol=2e-3 if dtype == torch.bfloat16 else 1e-7,
                               rtol=2e-2 if dtype == torch.bfloat16 else 1e-5)
    generated = model.generate(ids[:, :3], attention_mask=mask[:, :3], max_new_tokens=3,
                               do_sample=False, eos_token_id=None, use_cache=True)
    uncached = model.generate(ids[:, :3], attention_mask=mask[:, :3], max_new_tokens=3,
                              do_sample=False, eos_token_id=None, use_cache=False)
    assert torch.equal(generated, uncached)


def test_sdpa_attention_output_fallback_preserves_upstream_window_boundary():
    model = tiny_model(attn_implementation="sdpa", sliding_window=2)
    ids = torch.tensor([[1, 4, 8, 3, 7, 9]])
    mask, positions = torch.ones_like(ids), torch.arange(ids.shape[1])[None]
    eager = model(ids, attention_mask=mask, output_attentions=True, use_cache=False)
    expected, _, _ = reference_forward(model, ids, mask, positions, window=2)
    torch.testing.assert_close(eager.logits, expected, atol=1e-7, rtol=1e-5)
    # The upstream eager window includes the query and two preceding keys.
    assert torch.count_nonzero(eager.attentions[0][0, 0, -1]) == 3
    sdpa = model(ids, attention_mask=mask, use_cache=False)
    assert not torch.allclose(sdpa.logits[:, -1], eager.logits[:, -1])


def test_generation_config_keeps_full_cache_despite_sliding_window_metadata():
    model = tiny_model(sliding_window=2, layer_types=["sliding_attention"] * 2)
    assert model.config.layer_types == ["full_attention"] * 2
    ids = torch.tensor([[1, 4, 8, 3]])
    cached = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
    uncached = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None, use_cache=False)
    assert torch.equal(cached, uncached)


def test_explicit_additive_attention_mask_and_inputs_embeds_match_padding_mask():
    model = tiny_model(attn_implementation="sdpa")
    ids = torch.tensor([[1, 4, 8, 3], [0, 1, 5, 7]])
    mask = ids.ne(0)
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None] & mask[:, None, None]
    additive = torch.zeros(2, 1, 4, 4).masked_fill(~allowed, torch.finfo(torch.float32).min)
    ordinary = model(ids, attention_mask=mask, use_cache=False).logits
    explicit = model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=additive,
                     use_cache=False).logits
    torch.testing.assert_close(explicit, ordinary, atol=0, rtol=0)


def test_failed_mask_validation_preserves_cache_and_retry_matches_fresh():
    model = tiny_model()
    ids = torch.tensor([[1, 4, 8, 3, 7]])
    cache = model(ids[:, :3]).past_key_values
    before = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    with pytest.raises(ValueError, match="complete KV length"):
        model(ids[:, 3:], attention_mask=torch.ones(1, 2), past_key_values=cache)
    assert cache.get_seq_length() == 3
    for layer, (key, value) in zip(cache.layers, before):
        assert torch.equal(layer.keys, key) and torch.equal(layer.values, value)
    retried = model(ids[:, 3:], attention_mask=torch.ones(1, 5), past_key_values=cache).logits
    torch.testing.assert_close(retried, model(ids, use_cache=False).logits[:, 3:], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("tied", [False, True])
def test_safetensors_reload_resave_and_initialized_state_dict_load(tmp_path, tied):
    model = tiny_model(tie_word_embeddings=tied)
    ids = torch.tensor([[1, 5, 9, 3]])
    reference = model(ids).logits
    for index in range(2):
        path = tmp_path / str(index)
        model.save_pretrained(path)
        assert (path / "model.safetensors").is_file()
        assert not list(path.glob("*.bin")) and not list(path.glob("*.pt"))
        model = GrinMoeQModel.loader.from_pretrained(path)
        torch.testing.assert_close(model(ids).logits, reference, atol=0, rtol=0)
    for assign in (False, True):
        target = tiny_model(tie_word_embeddings=tied)
        with torch.no_grad():
            next(target.parameters()).add_(0.3)
        target.load_state_dict(model.state_dict(), assign=assign)
        torch.testing.assert_close(target(ids).logits, reference, atol=0, rtol=0)


def test_training_sparsemixer_gradients_and_tuple_outputs():
    model = tiny_model().train()
    torch.manual_seed(52)
    ids = torch.tensor([[1, 3, 7, 9], [1, 4, 8, 2]])
    result = model(ids, labels=ids, output_router_logits=True, use_cache=False)
    result.loss.backward()
    gradients = [layer.block_sparse_moe.gate.weight.grad for layer in model.model.layers]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    model.eval()
    result = model(ids, output_attentions=True, use_cache=False, return_dict=False)
    assert result[0].shape == (2, 4, 32)


def test_invalid_architecture_and_unsupported_cache_fail_explicitly():
    for kwargs in ({"num_experts_per_tok": 1}, {"num_local_experts": 1},
                   {"num_key_value_heads": 3}, {"hidden_size": 12}):
        with pytest.raises(ValueError):
            tiny_config(**kwargs)
    model = tiny_model()
    with pytest.raises(ValueError, match="exactly one"):
        model(torch.ones(1, 2, dtype=torch.long), inputs_embeds=torch.ones(1, 2, 16))
    with pytest.raises(ValueError, match="DynamicCache"):
        model(torch.ones(1, 2, dtype=torch.long), past_key_values=StaticCache(model.config, max_cache_len=8))


def test_no_exact_native_grinmoe_or_global_registration():
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    assert "grinmoe" not in CONFIG_MAPPING_NAMES
    assert "grinmoe" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


@pytest.mark.parametrize("fallback", [None, {}])
def test_awq_scales_quantizes_and_packs_every_projection_with_router_compensation(tmp_path, fallback):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.backend import BACKEND

    poisoned_checkpoint(tmp_path, hidden_size=32, intermediate_size=64, num_hidden_layers=1)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback, offload_to_disk=False)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH,
                             dtype=torch.float16, trust_remote_code=False)
    norm = wrapper.model.model.layers[0].post_attention_layernorm
    norm_weight, norm_bias = norm.weight.detach().float().clone(), norm.bias.detach().float().clone()
    attn_norm_weight = wrapper.model.model.layers[0].input_layernorm.weight.detach().clone()
    router = wrapper.model.model.layers[0].block_sparse_moe.gate.weight.detach().float().clone()
    calibration = [
        {"input_ids": torch.tensor([[1, 4, 7, 8, 9, 5, 3, 6]]),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.tensor([[1, 8, 6, 4, 3, 7, 9, 5]]),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
    ]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected = {"self_attn." + name + "_proj" for name in ("q", "k", "v", "o")}
    expected |= {f"block_sparse_moe.experts.{i}.{name}" for i in range(3) for name in ("w1", "w2", "w3")}
    layer = wrapper.model.model.layers[0]
    packed = {name for name, module in layer.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == expected
    assert all(module.qweight.dtype == torch.int32 for module in layer.modules() if isinstance(module, AwqTorchLinear))
    # Scale search ran: both LayerNorms absorbed a non-unit scale. The router
    # reads the post-attention norm output, so its input columns carry the
    # inverse scale and the routing logits are unchanged.
    assert not torch.equal(layer.input_layernorm.weight, attn_norm_weight)
    scale = norm_weight / layer.post_attention_layernorm.weight.detach().float()
    assert not torch.allclose(scale, torch.ones_like(scale))
    torch.testing.assert_close(layer.post_attention_layernorm.bias.float() * scale, norm_bias, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(layer.block_sparse_moe.gate.weight.float(), router * scale, atol=2e-3, rtol=2e-3)
    with torch.inference_mode():
        logits = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(logits).all()
    saved = tmp_path / "quantized"
    wrapper.save(str(saved))
    assert list(saved.glob("*.safetensors"))
    assert not list(saved.glob("*.bin")) and not list(saved.glob("*.pt"))
    reloaded = GPTQModel.load(str(saved), device="cpu", backend=BACKEND.TORCH,
                             dtype=torch.float16, trust_remote_code=False)
    assert isinstance(reloaded.model, GRINMoEForCausalLM)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, logits, atol=0, rtol=0)
    for name in expected:
        original = layer.get_submodule(name)
        restored = reloaded.model.model.layers[0].get_submodule(name)
        for key in ("qweight", "qzeros", "scales"):
            torch.testing.assert_close(getattr(restored, key), getattr(original, key), atol=0, rtol=0)
