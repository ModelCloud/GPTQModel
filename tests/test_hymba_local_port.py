# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the definition-owned NVIDIA Hymba implementation."""

import json
import threading

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import Cache, CacheLayerMixin

from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.hymba.configuration_hymba import HymbaConfig
from gptqmodel.models.ports.hymba.modeling_hymba import (
    HybridMambaAttentionDynamicCache,
    HymbaForCausalLM,
    selective_scan_reference,
)
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


def tiny_config(**overrides):
    values = {"vocab_size": 37, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 3,
              "num_attention_heads": 2, "num_key_value_heads": 1, "mamba_d_state": 3,
              "mamba_d_conv": 4, "mamba_dt_rank": 2, "num_experts": 1, "num_experts_per_tok": 1,
              "num_memory_tokens": 2, "kv_reuse_group": [[0, 1]], "rope": True,
              "sliding_window": 3, "global_attn_idx": [2], "eos_token_id": None}
    values.update(overrides)
    return HymbaConfig(**values)


def tiny_model(**overrides):
    torch.manual_seed(915)
    model = HymbaForCausalLM(tiny_config(**overrides)).eval()
    # Nonzero dt bias and nonuniform norm gain expose missing bias, row-order
    # errors, reset-state continuation, and normalization/scaling mistakes.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("dt_proj.0.bias"):
                param.copy_(torch.linspace(-0.5, 0.4, param.numel()))
            elif "layernorm" in name and name.endswith("weight"):
                param.copy_(torch.linspace(0.7, 1.3, param.numel()).reshape_as(param))
    return model


def _malicious_checkpoint(path, model):
    model.save_pretrained(path)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "evil_config.EvilConfig", "AutoModelForCausalLM": "evil_model.EvilModel"}
    config_path.write_text(json.dumps(config))
    marker = path / "EXECUTED"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint Python ran')\n"
    (path / "evil_config.py").write_text(code)
    (path / "evil_model.py").write_text(code)
    return marker


def test_definition_owns_selection_without_remote_trust():
    assert isinstance(HymbaQModel.loader, LocalModelLoader)
    assert not HymbaQModel.require_trust_remote_code


def test_definition_declares_actual_checkpoint_ssm_owners():
    from gptqmodel.models.module_plan import ModulePlan

    plan = ModulePlan.from_tree(HymbaQModel.module_tree)
    for owner in ("mamba.A_log", "mamba.D", "mamba.conv1d", "mamba.x_proj.0",
                  "mamba.dt_proj.0", "mamba.pre_avg_layernorm1", "mamba.pre_avg_layernorm2",
                  "pre_moe_layernorm"):
        assert plan.find(owner) is not None, owner


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.no_grad()
def test_actual_loader_ignores_malicious_auto_map(tmp_path, monkeypatch, trust, bypass):
    marker = _malicious_checkpoint(tmp_path, tiny_model())
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")  # Existing security API under test.
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is HymbaConfig
    with torch.device("meta"):
        shell = HymbaQModel.loader.from_config(config, trust_remote_code=trust, code_revision="checkpoint-code")
    assert type(shell) is HymbaForCausalLM
    assert all(param.device.type == "meta" for param in shell.parameters())
    loaded = HymbaQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust, code_revision="checkpoint-code")
    assert type(loaded) is HymbaForCausalLM
    ids = torch.tensor([[1, 8, 12, 9, 15]])
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, tiny_model()(ids, use_cache=False).logits,
                               rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.no_grad()
def test_public_gptqmodel_consumer_uses_real_local_model(tmp_path, monkeypatch, trust, bypass):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel

    original = tiny_model()
    marker = _malicious_checkpoint(tmp_path, original)
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")  # Existing security API under test.
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"t{i}": i for i in range(37)}, unk_token="t0")), unk_token="t0", pad_token="t0")
    tokenizer.save_pretrained(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert isinstance(wrapper, HymbaQModel)
    assert type(wrapper.model) is HymbaForCausalLM
    assert wrapper.module_plan().find("mamba.conv1d").quantize is False
    ids = torch.tensor([[1, 7, 8, 6]])
    torch.testing.assert_close(wrapper.model(ids, use_cache=False).logits, original(ids, use_cache=False).logits,
                               rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("length", [1, 3, 9])
@torch.no_grad()
def test_scan_matches_separate_float64_scalar_recurrence_with_nonzero_state(length):
    torch.manual_seed(44)
    u, delta, z = [torch.randn(2, length, 4) for _ in range(3)]
    A = -torch.rand(4, 3) - 0.1
    B, C = [torch.randn(2, length, 3) for _ in range(2)]
    D, initial = torch.randn(4), torch.randn(2, 4, 3)
    expected = torch.zeros_like(u, dtype=torch.float64)
    state = initial.double().clone()
    # Scalar loops deliberately avoid the production broadcasting expression.
    for b in range(2):
        for t in range(length):
            for i in range(4):
                dt = torch.log1p(torch.exp(delta[b, t, i].double()))
                for n in range(3):
                    state[b, i, n] = (state[b, i, n] * torch.exp(dt * A[i, n].double())
                                      + dt * B[b, t, n].double() * u[b, t, i].double())
                y = sum(state[b, i, n] * C[b, t, n].double() for n in range(3))
                y += D[i].double() * u[b, t, i].double()
                expected[b, t, i] = y * z[b, t, i].double().sigmoid() * z[b, t, i].double()
    actual, last = selective_scan_reference(u, delta, A, B, C, D, z, initial)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(last.double(), state, rtol=2e-5, atol=3e-6)


def _full_float64_oracle(model, input_ids):
    """Independent functional transcription of NVIDIA's full hybrid equations.

    Uses a state_dict, explicit causal softmax and convolution sums, FP64 scan,
    and no production forward, rotary, cache or norm helpers. This covers the
    published prefix-memory layout; interspersed insertion has separate tests.
    """
    c, weights = model.config, {key: value.double() for key, value in model.state_dict().items()}

    def linear(x, key):
        return F.linear(x, weights[key + ".weight"], weights.get(key + ".bias"))

    def norm(x, key, eps=None):
        return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + (eps or c.rms_norm_eps)) * weights[key + ".weight"]

    hidden = F.embedding(input_ids, weights["model.embed_tokens.weight"])
    mem = c.num_memory_tokens
    if mem:
        hidden = torch.cat((weights["model.memory_tokens"][None].expand(hidden.shape[0], -1, -1), hidden), 1)
    batch, length, _ = hidden.shape
    inner = int(c.hidden_size * c.mamba_expand)
    head = c.hidden_size // c.num_attention_heads
    vhead = inner // c.num_attention_heads
    key_width, value_width = head * c.num_key_value_heads, vhead * c.num_key_value_heads
    shared = {}
    for i in range(c.num_hidden_layers):
        root = f"model.layers.{i}"
        mixer = root + ".mamba"
        source = c.kv_sources[i]
        packed = linear(norm(hidden, root + ".input_layernorm"), mixer + ".in_proj")
        q = packed[..., :c.hidden_size].reshape(batch, length, c.num_attention_heads, head).transpose(1, 2)
        cursor = c.hidden_size
        if source == i:
            k = packed[..., cursor:cursor + key_width].reshape(batch, length, c.num_key_value_heads, head).transpose(1, 2)
            cursor += key_width
            v = packed[..., cursor:cursor + value_width].reshape(batch, length, c.num_key_value_heads, vhead).transpose(1, 2)
            cursor += value_width
            if c.kq_norm in ("rms", "perhead-rms"):
                k = norm(k, mixer + ".self_attn.k_norm", 1e-6)
        if c.kq_norm in ("rms", "perhead-rms"):
            q = norm(q, mixer + ".self_attn.q_norm", 1e-6)
        if c.rope:
            base = c.rope_theta
            if c.rope_type == "ntk":
                base *= (2 * c.max_position_embeddings / c.orig_max_position_embeddings - 1) ** (head / (head - 2))
            elif c.rope_type == "dynamic_ntk" and length > c.orig_max_position_embeddings:
                base *= (2 * length / c.orig_max_position_embeddings - 1) ** (head / (head - 2))
            angles = torch.arange(length).double()[:, None] * base ** (-torch.arange(0, head, 2).double() / head)
            angles = torch.cat((angles, angles), -1)[None, None]
            q = q * angles.cos() + torch.cat((-q[..., head // 2:], q[..., :head // 2]), -1) * angles.sin()
            if source == i:
                k = k * angles.cos() + torch.cat((-k[..., head // 2:], k[..., :head // 2]), -1) * angles.sin()
        if source != i:
            k, v = shared[source]
        shared[i] = k, v
        k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
        v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
        mask = torch.zeros(length, length, dtype=torch.bool)
        for query in range(length):
            for key in range(length):
                mask[query, key] = key <= query and (c.sliding_window is None or i in c.global_attn_idx
                                                    or query - key <= c.sliding_window or key < mem)
        scores = torch.matmul(q, k.transpose(-1, -2)) / head ** 0.5
        attn = torch.matmul(scores.masked_fill(~mask, -torch.inf).softmax(-1), v).transpose(1, 2).reshape(batch, length, inner)
        raw, gate = packed[..., cursor:cursor + inner], packed[..., cursor + inner:]
        convolved = torch.zeros_like(raw)
        for t in range(length):
            for lag in range(c.mamba_d_conv):
                if t + lag >= c.mamba_d_conv - 1:
                    convolved[:, t] += raw[:, t + lag - c.mamba_d_conv + 1] * weights[mixer + ".conv1d.weight"][:, 0, lag]
        if mixer + ".conv1d.bias" in weights:
            convolved += weights[mixer + ".conv1d.bias"]
        x = ACT(c.hidden_act, convolved)
        parameters = linear(x, mixer + ".x_proj.0")
        dt, B, C = parameters.split((c.mamba_dt_rank, c.mamba_d_state, c.mamba_d_state), -1)
        if c.mamba_inner_layernorms:
            dt, B, C = (norm(dt, mixer + ".dt_layernorm"), norm(B, mixer + ".B_layernorm"),
                         norm(C, mixer + ".C_layernorm"))
        dt = torch.log1p(linear(dt, mixer + ".dt_proj.0").exp())
        state = torch.zeros(batch, inner, c.mamba_d_state, dtype=torch.float64)
        ssm = torch.zeros_like(x)
        for t in range(length):
            state = state * torch.exp(-weights[mixer + ".A_log.0"].exp() * dt[:, t, :, None])
            state += dt[:, t, :, None] * x[:, t, :, None] * B[:, t, None]
            ssm[:, t] = (torch.einsum("bin,bn->bi", state, C[:, t]) + x[:, t] * weights[mixer + ".D.0"]) * F.silu(gate[:, t])
        hidden += linear((norm(attn, mixer + ".pre_avg_layernorm1") + norm(ssm, mixer + ".pre_avg_layernorm2")) / 2,
                         mixer + ".out_proj")
        if c.intermediate_size:
            x = norm(hidden, root + ".pre_moe_layernorm")
            contributions = []
            for expert in range(c.num_experts):
                prefix = root + f".moe.experts.{expert}"
                up = linear(x, prefix + ".up_proj")
                up = F.silu(linear(x, prefix + ".gate_proj")) * up if c.mlp_hidden_act == "silu" else up.relu().square()
                contributions.append(linear(up, prefix + ".down_proj"))
            if c.num_experts == 1:
                hidden += contributions[0]
            else:
                probs = linear(x, root + ".moe.router").float().softmax(-1).double()
                p, selected = probs.topk(c.num_experts_per_tok, dim=-1)
                routing = torch.zeros_like(probs).scatter(-1, selected, p)
                hidden += sum(y * routing[..., expert, None] for expert, y in enumerate(contributions))
    return linear(norm(hidden, "model.final_layernorm")[:, mem:], "lm_head")


def ACT(name, x):
    from transformers.activations import ACT2FN
    return ACT2FN[name](x)


@pytest.mark.parametrize("options", [
    {}, {"kv_reuse_group": None, "num_memory_tokens": 0, "rope": False},
    {"kv_reuse_group": None, "kv_reuse_every_i_layer": 3, "mamba_inner_layernorms": False},
    {"kq_norm": "rms", "mamba_proj_bias": True, "mamba_conv_bias": False},
    {"kq_norm": "perhead-rms", "num_key_value_heads": 2, "mlp_hidden_act": "relu2"},
    {"num_experts": 3, "num_experts_per_tok": 2},
    {"intermediate_size": 0},
    {"rope_type": "ntk", "orig_max_position_embeddings": 16, "max_position_embeddings": 64},
    {"rope_type": "dynamic_ntk", "orig_max_position_embeddings": 4, "max_position_embeddings": 64},
])
@torch.no_grad()
def test_full_hybrid_computation_matches_independent_float64_oracle(options):
    model = tiny_model(**options)
    ids = torch.tensor([[1, 6, 8, 9, 4, 12], [1, 3, 19, 14, 8, 2]])
    expected = _full_float64_oracle(model, ids)
    actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("options", [
    {}, {"num_memory_tokens": 0}, {"kv_reuse_group": None},
    {"kv_reuse_group": None, "kv_reuse_every_i_layer": 3},
    {"memory_tokens_interspersed_every": 3}, {"sliding_window": None},
    {"num_experts": 3, "num_experts_per_tok": 2, "kq_norm": "rms"},
    {"mamba_d_conv": 1, "mamba_inner_layernorms": False},
])
@torch.no_grad()
def test_short_prefill_multitoken_and_decode_continuation_match_full_forward(options):
    model = tiny_model(**options)
    ids = torch.tensor([[1, 8, 6, 11, 4, 12, 9, 13, 15]])
    expected = model(ids, use_cache=False).logits
    cache = None
    chunks = []
    for start, stop in ((0, 1), (1, 4), (4, 5), (5, 9)):
        output = model(ids[:, start:stop], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        chunks.append(output.logits)
    actual = torch.cat(chunks, dim=1)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=4e-7)
    assert cache.user_seq_length == ids.shape[1]
    assert isinstance(cache, Cache)
    for i, source in enumerate(model.config.kv_sources):
        layer = cache.layers[i]
        assert isinstance(layer, CacheLayerMixin) == (source == i)
        assert (layer.get_seq_length() == cache.get_seq_length()) if source == i else True
        assert layer.conv_states[0].shape[-1] == model.config.mamba_d_conv
        assert layer.recurrent_states[0].abs().sum() > 0
    memory_bands = (ids.shape[1] - 1) // model.config.memory_tokens_interspersed_every + 1 if model.config.memory_tokens_interspersed_every else 1
    assert cache.get_seq_length() == ids.shape[1] + memory_bands * model.config.num_memory_tokens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_low_precision_continuation_retains_dtype_and_fp32_oracle(dtype):
    base = tiny_model()
    model = tiny_model().to(dtype)
    ids = torch.tensor([[1, 8, 6, 11, 4]])
    expected = base(ids, use_cache=False).logits
    prefill = model(ids[:, :2], use_cache=True)
    output = model(ids[:, 2:], past_key_values=prefill.past_key_values, use_cache=True)
    assert output.past_key_values.layers[0].conv_states[0].dtype == dtype
    torch.testing.assert_close(output.logits, expected[:, 2:], rtol=0.08, atol=0.004)


@torch.no_grad()
def test_generate_greedy_and_beam_cache_reordering():
    model = tiny_model()
    ids = torch.tensor([[1, 8, 6, 11]])
    expected = ids
    for _ in range(3):
        token = model(expected, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, token), dim=-1)
    actual = model.generate(ids, max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    beams = model.generate(ids, max_new_tokens=3, do_sample=False, num_beams=2)
    assert beams.shape == expected.shape
    cache = model(torch.cat((ids, ids + 1), 0), use_cache=True).past_key_values
    def states():
        return [getattr(layer, name)[0] if name.endswith("states") else getattr(layer, name)
                for layer in cache.layers for name in ("keys", "values", "conv_states", "recurrent_states")
                if hasattr(layer, name)]
    saved = [value.clone() for value in states()]
    indices = torch.tensor([1, 0, 1])
    cache.reorder_cache(indices)
    assert len(saved) == len(states()) > 0
    for value, old in zip(states(), saved):
        torch.testing.assert_close(value, old[indices], rtol=0, atol=0)
    assert cache.batch_size == 3


@pytest.mark.parametrize("failure", ["layer", "head", "loss"])
@torch.no_grad()
def test_partial_failure_invalidates_cache_and_reset_allows_exact_retry(monkeypatch, failure):
    model = tiny_model()
    ids = torch.tensor([[1, 8, 6, 11, 4]])
    cache = model(ids[:, :2], use_cache=True).past_key_values
    old_state = cache.layers[0].recurrent_states[0].clone()
    if failure in ("layer", "head"):
        target = model.model.layers[1] if failure == "layer" else model.lm_head
        original = target.forward
        def fail(*args, **kwargs):
            raise RuntimeError("injected failure after state mutation")
        monkeypatch.setattr(target, "forward", fail)
        with pytest.raises(RuntimeError, match="injected failure"):
            model(ids[:, 2:], past_key_values=cache, use_cache=True)
        monkeypatch.setattr(target, "forward", original)
    else:
        with pytest.raises((IndexError, RuntimeError)):
            model(ids[:, 2:], labels=torch.full_like(ids[:, 2:], 999), past_key_values=cache, use_cache=True)
    assert not torch.equal(old_state, cache.layers[0].recurrent_states[0]), "must mutate a real prior layer before failing"
    with pytest.raises(RuntimeError, match="incomplete.*reset"):
        model(ids[:, 2:], past_key_values=cache, use_cache=True)
    cache.reset()
    retried = model(ids, past_key_values=cache, use_cache=True).logits
    torch.testing.assert_close(retried, model(ids, use_cache=False).logits, rtol=0, atol=0)


@pytest.mark.parametrize("tied", [False, True])
@torch.no_grad()
def test_safetensors_save_reload_resave_equivalence(tmp_path, tied):
    model = tiny_model(tie_word_embeddings=tied)
    ids = torch.tensor([[1, 3, 7, 19, 6]])
    expected = model(ids, use_cache=False).logits
    for index in range(2):
        destination = tmp_path / str(index)
        model.save_pretrained(destination, max_shard_size="5KB")
        assert list(destination.glob("*.safetensors"))
        assert not list(destination.glob("*.bin")) and not list(destination.glob("*.pt"))
        loaded = HymbaQModel.loader.from_pretrained(destination)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tied
        model = loaded


@pytest.mark.parametrize("options", [
    {"kv_reuse_group": [[1, 0]]}, {"kv_reuse_group": [[0, 1], [1, 2]]},
    {"kv_reuse_group": [[0, 3]]}, {"num_attention_heads": 3}, {"v_head_dim": 3},
    {"conv_dim": {"0": 99}}, {"rope_type": "ntk", "orig_max_position_embeddings": None},
])
def test_invalid_layout_is_rejected_before_allocation(options):
    with pytest.raises(ValueError):
        tiny_config(**options)


@pytest.mark.parametrize("options", [{}, {"num_memory_tokens": 0}, {"memory_tokens_interspersed_every": 3}])
@torch.no_grad()
def test_padding_causality_and_inputs_embeds(options):
    model = tiny_model(**options)
    ids = torch.tensor([[0, 0, 1, 8, 4, 7, 11], [0, 1, 2, 3, 6, 12, 9]])
    mask = ids.ne(0)
    expected = model(ids, attention_mask=mask, use_cache=False).logits
    assert torch.isfinite(expected).all()
    embedded = model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(expected, embedded, rtol=0, atol=0)
    first = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    second = model(ids[:, 3:], attention_mask=mask, past_key_values=first.past_key_values, use_cache=True)
    torch.testing.assert_close(second.logits, expected[:, 3:], rtol=5e-5, atol=4e-7)
    changed = ids.clone()
    changed[:, -2:] = torch.tensor([[20, 21], [22, 23]])
    perturbed = model(changed, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(perturbed[:, :-2], expected[:, :-2], rtol=0, atol=0)


@torch.no_grad()
def test_cache_rejects_cross_request_ownership_and_unsupported_rollback():
    model, other = tiny_model(), tiny_model()
    ids = torch.tensor([[1, 8, 9]])
    cache = model(ids, use_cache=True).past_key_values
    with pytest.raises(ValueError, match="different model"):
        other(ids, past_key_values=cache, use_cache=True)
    cache.reset()
    other(ids, past_key_values=cache, use_cache=True)
    with pytest.raises(NotImplementedError, match="cannot be cropped"):
        cache.crop(1)
    assert torch.isfinite(other(ids, past_key_values=cache, use_cache=True).logits).all()


def test_cache_rejects_other_thread_before_any_tensor_access():
    cache = HybridMambaAttentionDynamicCache(tiny_config(), batch_size=1)
    errors = []
    def second_request():
        try:
            with cache.transaction():
                errors.append("entered")
        except RuntimeError as exc:
            errors.append(str(exc))
    with cache.transaction():
        thread = threading.Thread(target=second_request)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(errors) == 1 and "concurrently" in errors[0]
    # A rejected second request cannot poison the first request's cache.
    with cache.transaction():
        pass


@torch.no_grad()
def test_no_mlp_router_output_does_not_invent_experts_or_break_continuation():
    model = tiny_model(intermediate_size=0)
    ids = torch.tensor([[1, 8, 6, 11, 4]])
    expected = model(ids, use_cache=False).logits
    first = model(ids[:, :2], use_cache=True, output_router_logits=True)
    second = model(ids[:, 2:], past_key_values=first.past_key_values,
                   use_cache=True, output_router_logits=True)
    assert first.router_logits == second.router_logits == ()
    assert first.aux_loss is second.aux_loss is None
    torch.testing.assert_close(second.logits, expected[:, 2:], rtol=3e-5, atol=4e-7)


@pytest.mark.parametrize("options", [
    {}, {"num_experts": 3, "num_experts_per_tok": 2, "kq_norm": "perhead-rms"},
    {"mlp_hidden_act": "relu2", "mamba_inner_layernorms": False},
    {"intermediate_size": 0},
])
@torch.no_grad()
def test_awq_actual_plain_fallback_consumes_all_declared_projections_without_folding(options):
    from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
    from gptqmodel.looper.loop_processor import CalibrationConfig
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization.config import FORMAT, AWQConfig

    model = tiny_model(**options)
    wrapper = HymbaQModel(model=model, quantized=False, quantize_config=None)
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(format=FORMAT.GEMM, group_size=4, sym=False),
                             calibration=CalibrationConfig(), gptq_model=wrapper,
                             model=model, calculate_w_wq_diff=False)
    # Observe the boundary but run the real affine quantizer/staging consumer.
    # GPU packing is deliberately outside this CPU qualification.
    applied = []
    real_apply_quant = processor.apply_quant
    def apply_plain(named_linears, scales_list):
        assert scales_list == []
        for name, named in named_linears.items():
            assert isinstance(named.module, torch.nn.Linear)
            applied.append(name)
        real_apply_quant(named_linears, scales_list)
        for named in named_linears.values():
            assert torch.isfinite(named.state["wq"]).all()
            assert torch.isfinite(named.state["q_scales"]).all()
            assert torch.isfinite(named.state["q_zeros"]).all()
    processor.apply_quant = apply_plain
    ids = torch.tensor([[1, 8, 6, 11, 4]])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    quantized_keys = set()
    for index, layer in enumerate(model.model.layers):
        targets = {name: module for name, module in layer.named_modules()
                   if isinstance(module, torch.nn.Linear) and wrapper.module_plan().find(name).quantize}
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        state = _AWQLayerState(
            modules={name: NamedModule(module, name=name, full_name=f"model.layers.{index}.{name}", layer_index=index)
                     for name, module in targets.items()},
            layer_module=layer, pending_modules=set(targets), subset_total=1, processed_subsets={0},
        )
        processor._quantize_layer(index, state)
        assert state.quantized and state.modules == {} and state.layer_module is None
        assert state.pending_modules == set()
        quantized_keys.update(f"model.layers.{index}.{name}.weight" for name in targets)
    assert len(applied) == len(quantized_keys) > 0
    for name, value in model.state_dict().items():
        if name in quantized_keys:
            assert not torch.equal(value, before[name]), name
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    logits = model(ids, use_cache=False).logits
    assert logits.shape == (1, 5, 37) and torch.isfinite(logits).all()
    # Keep mutable replay context empty; plain quantization needs no candidate
    # replay or norm/SSM/router compensation at all.
    assert processor._scale_feature_by_module == {}
    assert isinstance(processor._module_forward_kwargs, dict)


@pytest.mark.parametrize("sources", [[[0, 1, 2]], [[0, 2]]])
@torch.no_grad()
def test_actual_calibration_worker_preserves_native_layer_masks_and_shared_kv(sources):
    from types import SimpleNamespace

    from gptqmodel.utils.looper_helpers import forward_batch_worker

    # Layer 0 is global; layer 1/2 are local. Capture exactly the first-layer
    # kwargs that StageInputsCapture replays, including its stale use_swa=False.
    model = tiny_model(kv_reuse_group=sources, global_attn_idx=[0])
    wrapper = HymbaQModel(model=model, quantized=False, quantize_config=None)
    ids = torch.tensor([[1, 8, 6, 11, 4, 7, 13], [1, 3, 9, 15, 20, 6, 4]])
    captured, native = {}, []
    def first_input(layer, args, kwargs):
        captured.update(hidden=args[0].clone(), kwargs=dict(kwargs))
    def layer_output(layer, args, output):
        native.append(output[0].clone())
    handles = [model.model.layers[0].register_forward_pre_hook(first_input, with_kwargs=True)]
    handles.extend(layer.register_forward_hook(layer_output) for layer in model.model.layers)
    model(ids, use_cache=False)
    for handle in handles:
        handle.remove()
    # Each calibration batch keeps its own replay context. Exercise the real
    # worker, wrapper hooks and native layer modules; only progress bookkeeping
    # is a minimal stand-in for a processor's batch-index setter.
    for batch in range(2):
        kwargs = {key: value[batch:batch + 1] if torch.is_tensor(value) else value
                  for key, value in captured["kwargs"].items()}
        attention, positions = kwargs.pop("attention_mask"), kwargs.pop("position_ids")
        hidden, previous = captured["hidden"][batch:batch + 1], None
        processor = SimpleNamespace(_set_current_batch_index=lambda index: None)
        for i, layer in enumerate(model.model.layers):
            _, hidden, previous = forward_batch_worker(
                layer, processor, batch, [hidden], kwargs, attention, positions,
                gptq_model=wrapper, support_batch_quantize=True, is_lm_head_module=False,
                need_output=True, reuse_kv=layer.reuse_kv, prev_kv=previous,
                write_shared_kv_cache=wrapper.write_shared_kv_cache,
            )
            torch.testing.assert_close(hidden, native[i][batch:batch + 1], rtol=2e-5, atol=2e-6)
        bank = kwargs.get("_hymba_kv_sources", {})
        assert set(bank) <= {0}  # the only source with a future consumer


@torch.no_grad()
def test_recorded_outputs_and_tuple_return():
    model = tiny_model(num_experts=3, num_experts_per_tok=2)
    ids = torch.tensor([[1, 8, 6, 11]])
    output = model(ids, use_cache=False, output_hidden_states=True, output_router_logits=True)
    layers = model.config.num_hidden_layers
    assert len(output.hidden_states) == layers + 1 and len(output.router_logits) == layers
    torch.testing.assert_close(output.hidden_states[-1], model.model(ids, use_cache=False).last_hidden_state)
    as_tuple = model(ids, use_cache=False, return_dict=False)
    assert isinstance(as_tuple, tuple)
    torch.testing.assert_close(as_tuple[0], output.logits, rtol=0, atol=0)
