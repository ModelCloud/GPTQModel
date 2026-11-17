# SPDX-License-Identifier: Apache-2.0
"""Real LLaDA2 diffusion checkpoints, not a causal model with similar shapes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from gptqmodel.models.definitions.llada2 import LLaDA2MoeQModel
from gptqmodel.models.ports.llada2.configuration_llada2_moe import LLaDA2MoeConfig
from gptqmodel.models.ports.llada2.modeling_llada2_moe import LLaDA2MoeModelLM
from gptqmodel.quantization.config import CalibrationConfig
from gptqmodel.utils.hf import load_model_config


VERSIONS = ["2.0", "2.0-mini-preview", "2.0-flash-preview", "2.0-cap", "2.1", "2.2"]
AUDITS = json.loads((Path(__file__).parents[1] / "gptqmodel/models/ports/llada2/checkpoint_audits.json").read_text())


def tiny_config(version="2.1", **overrides):
    values = {
        "vocab_size": 37,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "n_group": 2,
        "topk_group": 1,
        "moe_intermediate_size": 16,
        "first_k_dense_replace": 1,
        "pad_token_id": 0,
        "max_position_embeddings": 32,
        "embedding_dropout": 0,
        "attention_dropout": 0,
        "output_dropout": 0,
        "llada_version": version,
        "use_cache": False,
        "attn_implementation": "eager",
    }
    if version == "2.2":
        values.update(block_size=2, expert_capacity=3, delete_token_id=33, split_token_id=34)
    values.update(overrides)
    return LLaDA2MoeConfig(**values)


def tiny_model(version="2.1", **overrides):
    torch.manual_seed(342)
    model = LLaDA2MoeModelLM(tiny_config(version, **overrides)).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.expert_bias.copy_(torch.tensor([0.09, -0.08, 0.13, -0.04]))
    return model


def hostile_checkpoint(path, version="2.1"):
    model = tiny_model(version)
    model.save_pretrained(path, safe_serialization=True)
    config = json.loads((path / "config.json").read_text())
    config["auto_map"] = dict.fromkeys(
        ("AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoTokenizer"), "evil.CheckpointCode"
    )
    (path / "config.json").write_text(json.dumps(config))
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    return model, marker


def test_loading_does_not_mutate_any_forward_class(monkeypatch):
    model = tiny_model()
    wrapper = object.__new__(LLaDA2MoeQModel)
    outer, inner = type(model).forward, type(model.model).forward
    monkeypatch.setattr(type(model), "forward", outer)
    monkeypatch.setattr(type(model.model), "forward", inner)
    wrapper.after_model_load(model, load_quantized_model=False)
    assert type(model).forward is outer, "loading one model must not patch every instance's forward"
    assert type(model.model).forward is inner


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_checkpoint_python_cannot_select_config_or_model(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original, marker = hostile_checkpoint(tmp_path)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is LLaDA2MoeConfig
    loaded = LLaDA2MoeQModel.loader.from_pretrained(
        tmp_path,
        config=config,
        trust_remote_code=trust,
        attn_implementation="eager",
    ).eval()
    assert type(loaded) is LLaDA2MoeModelLM
    ids = torch.tensor([[1, 5, 7, 8]])
    torch.testing.assert_close(loaded(ids).logits, original(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("name", list(AUDITS))
def test_exact_full_checkpoint_key_inventory_on_meta(name):
    audit = AUDITS[name]
    config = LLaDA2MoeConfig(**{**audit["config"], "llada_version": audit["llada_version"]})
    with torch.device("meta"):
        model = LLaDA2MoeModelLM(config)
    state = model.state_dict()
    assert len(state) == audit["key_count"]
    digest = hashlib.sha256("\n".join(sorted(state)).encode()).hexdigest()
    assert digest == audit["keys_sha256"]
    assert all(tensor.is_meta for tensor in state.values())
    assert state["model.layers.0.attention.query_key_value.weight"].shape == (
        (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
        config.hidden_size,
    )
    first_moe = config.first_k_dense_replace
    assert state[f"model.layers.{first_moe}.mlp.experts.0.gate_proj.weight"].shape == (
        config.moe_intermediate_size,
        config.hidden_size,
    )
    assert state[f"model.layers.{first_moe}.mlp.gate.expert_bias"].dtype == torch.float32


def reference_rms(value, weight, epsilon):
    return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon).to(value.dtype) * weight


def reference_forward(model, ids, visible=None):
    """Independent dense equations: no port forward, mask, RoPE or router helper."""
    cfg, state = model.config, model.state_dict()
    hidden = F.embedding(ids, state["model.word_embeddings.weight"])
    batch, length = ids.shape
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    frequencies = cfg.rope_theta ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    angles = torch.arange(length, dtype=torch.float64)[:, None] * frequencies[None, :]
    angles = torch.cat([angles, angles], -1).to(hidden.dtype)[None, None]

    def rope(value):
        rotating, rest = value[..., :rotary_dim], value[..., rotary_dim:]
        half = rotary_dim // 2
        rotated = torch.cat([-rotating[..., half:], rotating[..., :half]], -1)
        return torch.cat([rotating * angles.cos() + rotated * angles.sin(), rest], -1)

    def linear(value, prefix):
        return F.linear(value, state[prefix + ".weight"], state.get(prefix + ".bias"))

    def norm(value, prefix):
        return reference_rms(value, state[prefix + ".weight"], cfg.rms_norm_eps)

    def mlp(value, prefix):
        return linear(
            F.silu(linear(value, prefix + ".gate_proj")) * linear(value, prefix + ".up_proj"), prefix + ".down_proj"
        )

    hidden_trace, routers = [hidden], []
    for index in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{index}."
        normalized = norm(hidden, prefix + "input_layernorm")
        packed = linear(normalized, prefix + "attention.query_key_value")
        sizes = [cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim]
        query, key, value = packed.split([sizes[0], sizes[1], sizes[1]], -1)
        query = query.reshape(batch, length, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        key = key.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        if cfg.use_qk_norm:
            query = norm(query, prefix + "attention.query_layernorm")
            key = norm(key, prefix + "attention.key_layernorm")
        query, key = rope(query), rope(key)
        attended = []
        for head in range(cfg.num_attention_heads):
            kv = head // (cfg.num_attention_heads // cfg.num_key_value_heads)
            scores = query[:, head] @ key[:, kv].transpose(-1, -2) / cfg.head_dim**0.5
            if visible is not None:
                scores = scores.masked_fill(~visible[:, 0], torch.finfo(scores.dtype).min)
            probabilities = scores.float().softmax(-1).to(query.dtype)
            attended.append(probabilities @ value[:, kv])
        attended = torch.stack(attended, 2).flatten(2)
        hidden = hidden + linear(attended, prefix + "attention.dense")
        normalized = norm(hidden, prefix + "post_attention_layernorm")
        if cfg.num_experts is None or index < cfg.first_k_dense_replace:
            feedforward = mlp(normalized, prefix + "mlp")
        else:
            flat = normalized.flatten(0, 1)
            logits = F.linear(flat.float(), state[prefix + "mlp.gate.weight"].float())
            scores = logits.sigmoid()
            biased = scores + state[prefix + "mlp.gate.expert_bias"]
            selected = []
            for row in range(flat.shape[0]):
                if cfg.llada_version == "2.2":
                    begin = row // cfg.block_size * cfg.block_size
                    priorities = biased[begin : begin + cfg.block_size].amax(0)
                    allowed = priorities.topk(cfg.expert_capacity).indices.tolist()
                else:
                    groups = [
                        biased[
                            row,
                            group * (cfg.num_experts // cfg.n_group) : (group + 1) * (cfg.num_experts // cfg.n_group),
                        ]
                        for group in range(cfg.n_group)
                    ]
                    priorities = torch.stack([group.topk(2).values.sum() for group in groups])
                    chosen_groups = priorities.topk(cfg.topk_group).indices.tolist()
                    allowed = [
                        expert
                        for expert in range(cfg.num_experts)
                        if expert // (cfg.num_experts // cfg.n_group) in chosen_groups
                    ]
                eligible = torch.full_like(biased[row], -torch.inf)
                eligible[allowed] = biased[row, allowed]
                selected.append(eligible.topk(cfg.num_experts_per_tok).indices)
            selected = torch.stack(selected)
            weight = scores.gather(-1, selected)
            if cfg.num_experts_per_tok > 1:
                weight = weight / (weight.sum(-1, keepdim=True) + 1e-20)
            weight *= cfg.routed_scaling_factor
            feedforward = torch.zeros_like(flat)
            for row in range(flat.shape[0]):
                for slot, expert in enumerate(selected[row].tolist()):
                    feedforward[row] += (
                        weight[row, slot] * mlp(flat[row : row + 1], prefix + f"mlp.experts.{expert}")[0]
                    )
            if cfg.num_shared_experts:
                feedforward += mlp(flat, prefix + "mlp.shared_experts")
            feedforward = feedforward.view_as(hidden)
            routers.append((logits.view(batch, length, -1), selected.view(batch, length, -1)))
        hidden = hidden + feedforward
        if index != cfg.num_hidden_layers - 1:
            hidden_trace.append(hidden)
    hidden = norm(hidden, "model.norm")
    hidden_trace.append(hidden)
    return linear(hidden, "lm_head").float(), hidden_trace, routers


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kv_heads,bias,qk_norm", [(4, False, False), (2, True, True), (1, False, True)])
@pytest.mark.parametrize("mask_kind", ["bidirectional", "block", "causal"])
def test_entire_forward_independent_equations(version, kv_heads, bias, qk_norm, mask_kind):
    model = tiny_model(
        version, num_key_value_heads=kv_heads, use_qkv_bias=bias, use_bias=bias, use_qk_norm=qk_norm
    ).double()
    ids = torch.tensor([[1, 2, 7, 3], [4, 9, 8, 5]])
    positions = torch.arange(4)
    if mask_kind == "bidirectional":
        mask, visible = None, None
    elif mask_kind == "block":
        visible = ((positions // 2)[:, None] >= (positions // 2)[None, :])[None, None].expand(2, 1, 4, 4)
        mask = visible.float()
    else:
        visible = (positions[:, None] >= positions[None, :])[None, None].expand(2, 1, 4, 4)
        mask = torch.ones_like(ids)
    with torch.no_grad():
        expected, traces, routers = reference_forward(model, ids, visible)
        actual = model(
            ids, attention_mask=mask, output_hidden_states=True, output_router_logits=True, output_attentions=True
        )
    torch.testing.assert_close(actual.logits, expected, rtol=1e-5, atol=2e-7)
    for result, oracle in zip(actual.hidden_states, traces, strict=True):
        torch.testing.assert_close(result, oracle, rtol=1e-5, atol=2e-7)
    for result, oracle in zip(actual.router_logits, routers, strict=True):
        torch.testing.assert_close(result, oracle, rtol=1e-5, atol=2e-7)
    assert actual.attentions[0].shape == (2, 4, 4, 4)


@pytest.mark.parametrize("version", VERSIONS)
def test_real_diffusion_generation_against_independent_equations(version):
    model = tiny_model(version).double()
    prompt = torch.tensor([[1, 2, 3]])
    block, generated = 4, 5
    total = ((prompt.shape[1] + generated + block - 1) // block) * block
    expected = torch.full((1, total), 35)
    expected[:, :3] = prompt
    with torch.no_grad():
        for end in range(4, total + 1, 4):
            positions = torch.arange(end)
            visible = ((positions // block)[:, None] >= (positions // block)[None, :])[None, None]
            logits, _, _ = reference_forward(model, expected[:, :end], visible)
            candidates = logits.argmax(-1)
            active = expected[:, :end] == 35
            expected[:, :end][active] = candidates[active]
    model.config.use_cache = True
    result = model.generate(
        prompt,
        gen_length=generated,
        block_length=block,
        steps=2,
        mask_id=35,
        eos_id=None,
        threshold=-1,
        temperature=0,
        top_k=1,
        **({"max_post_steps": 0, "editing_threshold": 2} if version in ("2.1", "2.2") else {}),
    )
    expected = expected[:, :8] if version == "2.0-mini-preview" else expected[:, 3:8]
    assert torch.equal(result, expected)
    assert model.config.use_cache is True


@pytest.mark.parametrize("version", VERSIONS)
def test_native_sampler_matches_independent_probability_and_rng_oracle(version):
    model = tiny_model(version)
    logits = torch.tensor([[[0.5, 1.2, -0.8, 0.4], [0.1, 0.3, 0.8, 0.2]]])
    for temperature, top_k, top_p in [(0, None, None), (0.7, 3, 0.9), (1, 2, 1)]:
        flat = logits.flatten(0, 1).clone()
        if temperature > 0:
            flat /= temperature
        for row in flat:
            if top_k:
                allowed = row.argsort(descending=True)[:top_k]
                row[[index for index in range(4) if index not in allowed]] = -torch.inf
            if top_p is not None and top_p < 1:
                order = row.argsort(descending=True)
                cumulative = row[order].softmax(0).cumsum(0)
                for rank in range(1, 4):
                    if cumulative[rank - 1] > top_p:
                        row[order[rank]] = -torch.inf
        distribution = flat.softmax(-1)
        greedy = version in ("2.0-mini-preview", "2.1") and temperature == 0 and not top_k and top_p is None
        torch.manual_seed(123)
        expected = flat.argmax(-1, keepdim=True) if greedy else torch.multinomial(distribution, 1)
        confidence = distribution.gather(-1, expected).reshape(1, 2)
        torch.manual_seed(123)
        tokens, probability = model._sample_with_temperature_topk_topp(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        assert torch.equal(tokens, expected.reshape(1, 2))
        torch.testing.assert_close(probability, confidence)


@pytest.mark.parametrize("version", VERSIONS)
def test_eos_return_conventions_and_block_mask(version, monkeypatch):
    model = tiny_model(version)
    seen = []

    def scripted_forward(input_ids, attention_mask, position_ids, **kwargs):
        seen.append((input_ids.clone(), attention_mask.clone()))
        logits = torch.full((*input_ids.shape, 37), -50.0)
        logits[..., 36] = 50
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(model, "forward", scripted_forward)
    result = model.generate(
        torch.tensor([[1, 2, 3]]),
        gen_length=3,
        block_length=4,
        steps=2,
        mask_id=35,
        eos_id=36,
        eos_early_stop=True,
        top_k=1,
        **({"max_post_steps": 0} if version in ("2.1", "2.2") else {}),
    )
    expected = [[36]] if version in ("2.1", "2.2") else [[1, 2, 3, 36]]
    assert result.tolist() == expected
    assert seen[0][0].tolist() == [[1, 2, 3, 35]]
    # One block must be bidirectional, not the ordinary causal token triangle.
    assert seen[0][1].shape == (1, 1, 4, 4)
    assert bool((seen[0][1] == (1 if version == "2.1" else 0)).all())


def test_21_edits_non_prompt_tokens_after_denoising(monkeypatch):
    model = tiny_model("2.1")
    seen = []

    def edit_logits(input_ids, **kwargs):
        seen.append(input_ids.clone())
        logits = torch.full((*input_ids.shape, 37), -50.0)
        logits[..., 6 if len(seen) == 1 else 7] = 50
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(model, "forward", edit_logits)
    result = model.generate(
        torch.tensor([[1, 2, 3]]), gen_length=1, block_length=4, steps=1, mask_id=35, eos_id=None, max_post_steps=2
    )
    assert result.tolist() == [[7]]
    assert len(seen) == 3
    assert all(ids[0, :3].tolist() == [1, 2, 3] for ids in seen)
    assert [ids[0, 3].item() for ids in seen] == [35, 6, 7]


def test_22_delete_split_tracking_final_round_and_loop_escape(monkeypatch):
    model = tiny_model("2.2")
    edited, tracking = model._apply_edit_operations_with_tracking(
        [8, 33, 34, 9],
        [8, 2, 3, 9],
        [False, True, True, False],
        mask_id=35,
        block_length=4,
        delete_token_id=33,
        split_token_id=34,
    )
    assert edited == [8, 35, 3, 9]
    assert tracking == [False, False, True, False]
    seen = []

    def logits(input_ids, **kwargs):
        seen.append(input_ids.clone())
        scores = torch.full((*input_ids.shape, 37), -10.0)
        for pos, token in enumerate([1, 33, 34, 8] if len(seen) == 1 else [1, 6, 7, 8]):
            scores[0, pos, token] = 40 - pos
        return SimpleNamespace(logits=scores)

    monkeypatch.setattr(model, "forward", logits)
    result = model.generate(
        torch.tensor([[1]]),
        gen_length=3,
        block_length=4,
        steps=2,
        threshold=2,
        editing_threshold=2,
        mask_id=35,
        eos_id=None,
        max_post_steps=0,
    )
    assert result.tolist() == [[6, 7, 8]]
    assert seen[1].tolist() == [[1, 35, 35, 35]]
    # On the terminal refinement pass, edit tokens must be resampled away.
    seen.clear()
    result = model.generate(
        torch.tensor([[1]]),
        gen_length=3,
        block_length=4,
        steps=1,
        threshold=2,
        editing_threshold=2,
        mask_id=35,
        eos_id=None,
        max_post_steps=0,
    )
    assert not bool(((result == 33) | (result == 34) | (result == 35)).any())
    block = torch.tensor([4, 5])
    scores = torch.zeros(2, 37)
    scores[0, 6], scores[1, 7] = 10, 10
    model._resample_to_escape_loop(
        block,
        torch.tensor([1, 2]),
        scores,
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        {(4, 5)},
        temperature=0,
        top_k=None,
        top_p=None,
    )
    assert tuple(block.tolist()) not in {(4, 5)}


def write_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i for i in range(2, 37)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(path)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "trust,bypass", [(False, None), (True, None), (False, "0"), (True, "0"), (False, "1"), (True, "1")]
)
def test_public_explicit_version_and_hostile_maps(tmp_path, monkeypatch, version, trust, bypass):
    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original, marker = hostile_checkpoint(tmp_path, version)
    write_tokenizer(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    data.pop("llada_version")
    (tmp_path / "config.json").write_text(json.dumps(data))
    loaded = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust, llada_version=version
    )
    assert type(loaded) is LLaDA2MoeQModel
    assert type(loaded.model) is LLaDA2MoeModelLM
    assert loaded.model.config.llada_version == version
    assert not loaded.trust_remote_code
    assert not marker.exists()
    ids = torch.tensor([[2, 3, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits, rtol=1e-5, atol=2e-7)
    loaded.model.config.save_pretrained(tmp_path / "config_only")
    assert LLaDA2MoeQModel.loader.load_config(tmp_path / "config_only").llada_version == version


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_safetensors_dtype_roundtrip_and_failed_retry(tmp_path, version, dtype):
    from safetensors.torch import load_file, save_file

    model = tiny_model(version)
    # Check strict FP32 on load, not on .to(dtype), which intentionally casts all buffers.
    model.save_pretrained(tmp_path, safe_serialization=True)
    first = LLaDA2MoeQModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    assert first.model.layers[1].mlp.gate.expert_bias.dtype == torch.float32
    ids = torch.tensor([[2, 3, 5, 6]])
    expected = first(ids).logits
    assert torch.isfinite(expected).all()
    destination = tmp_path / "roundtrip"
    first.save_pretrained(destination, safe_serialization=True)
    second = LLaDA2MoeQModel.loader.from_pretrained(destination, dtype=dtype).eval()
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(second(ids).logits, expected, rtol=0, atol=0)
    assert second.config.llada_version == version
    assert not list(destination.glob("*.bin"))
    weights = load_file(destination / "model.safetensors")
    bad = dict(weights)
    bad["model.layers.1.attention.query_key_value.weight"] = torch.ones(1, 1)
    save_file(bad, destination / "model.safetensors")
    with pytest.raises((RuntimeError, ValueError)):
        LLaDA2MoeQModel.loader.from_pretrained(destination, dtype=dtype)
    save_file(weights, destination / "model.safetensors")
    retried = LLaDA2MoeQModel.loader.from_pretrained(destination, dtype=dtype).eval()
    torch.testing.assert_close(retried(ids).logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_causal_cache_continuation_invalid_mask_retry_and_diffusion_visibility(version, implementation):
    from transformers.cache_utils import DynamicCache

    model = tiny_model(version, attn_implementation=implementation)
    ids = torch.tensor([[2, 3, 5, 6]])
    with torch.no_grad():
        full = model(ids, attention_mask=torch.ones_like(ids)).logits
        prefill = model(ids[:, :2], attention_mask=torch.ones(1, 2), use_cache=True)
        cache = prefill.past_key_values
        assert isinstance(cache, DynamicCache)
        with pytest.raises(ValueError, match="shape"):
            model(ids[:, 2:], past_key_values=cache, attention_mask=torch.ones(1, 3), use_cache=True)
        assert cache.get_seq_length() == 2
        continued = model(ids[:, 2:], past_key_values=cache, attention_mask=torch.ones(1, 4), use_cache=True)
        torch.testing.assert_close(continued.logits, full[:, 2:], rtol=1e-5, atol=2e-7)
        assert cache.get_seq_length() == 4
        changed = ids.clone()
        changed[0, 3] = 9
        causal = model(changed, attention_mask=torch.ones_like(ids)).logits
        torch.testing.assert_close(causal[:, :3], full[:, :3], rtol=0, atol=0)
        block_mask = (torch.arange(4)[:, None] // 2 >= torch.arange(4)[None, :] // 2)[None, None].float()
        original = model(ids, attention_mask=block_mask).logits
        block_changed = model(changed, attention_mask=block_mask).logits
        torch.testing.assert_close(original[:, :2], block_changed[:, :2], rtol=0, atol=0)
        assert not torch.allclose(original[:, 2], block_changed[:, 2])
        # Additive zeros preserve full bidirectional attention; integer 2D ones are causal.
        bidirectional = model(ids).logits
        zeros = model(ids, attention_mask=torch.zeros(1, 1, 4, 4)).logits
        torch.testing.assert_close(zeros, bidirectional, rtol=0, atol=0)
        assert not torch.allclose(bidirectional[:, 0], full[:, 0])


@pytest.mark.parametrize(
    "version,kind",
    [
        ("2.0", "mixed"),
        ("2.0-mini-preview", "dense"),
        ("2.0-flash-preview", "moe"),
        ("2.0-cap", "mixed"),
        ("2.1", "mixed"),
        ("2.2", "mixed"),
    ],
)
@pytest.mark.parametrize("offload", [False, True])
def test_public_plain_awq_all_owners_pack_infer_and_two_reloads(tmp_path, version, kind, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    dense = {"dense": 2, "mixed": 1, "moe": 0}[kind]
    reference = tiny_model(version, first_k_dense_replace=dense, use_bias=False)
    with torch.no_grad():
        for layer in reference.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.weight.normal_(std=0.5)
                layer.mlp.gate.expert_bias.copy_(torch.tensor([0.001, -0.0008, 0.0013, -0.0004]))
    reference.save_pretrained(tmp_path, safe_serialization=True)
    write_tokenizer(tmp_path)
    # Force the same API-only dispatch path as ambiguous published configs.
    data = json.loads((tmp_path / "config.json").read_text())
    data.pop("llada_version")
    (tmp_path / "config.json").write_text(json.dumps(data))
    wrapper = GPTQModel.load(
        str(tmp_path),
        llada_version=version,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload),
    )
    assert wrapper.model.config.llada_version == version
    for layer in wrapper.model.model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    expected_names = {
        name for name, mod in wrapper.model.named_modules() if isinstance(mod, torch.nn.Linear) and name != "lm_head"
    }
    for layer in range(dense, 2):
        expected_names.discard(f"model.layers.{layer}.mlp.gate")
    calibration = [
        {
            "input_ids": (torch.arange(8) + start).remainder(34).view(1, 8) + 2,
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        for start in [0, 8, 16, 24]
    ]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    actual_names = {name for name, mod in wrapper.model.named_modules() if isinstance(mod, AwqTorchLinear)}
    assert actual_names == expected_names
    assert len(actual_names) == 4 + dense * 3 + (2 - dense) * 15
    for name in actual_names:
        quantized = wrapper.model.get_submodule(name)
        assert quantized.qweight.dtype == torch.int32
        assert not hasattr(quantized, "weight")
    ids = calibration[0]
    with torch.no_grad():
        expected = wrapper.model(**ids).logits
    assert torch.isfinite(expected).all()
    source_state = {name: value.clone() for name, value in wrapper.model.state_dict().items()}
    for cycle in range(2):
        output = tmp_path / f"quantized{cycle}"
        wrapper.save_quantized(str(output))
        assert list(output.glob("*.safetensors"))
        assert not list(output.glob("*.bin"))
        assert json.loads((output / "config.json").read_text())["llada_version"] == version
        wrapper = GPTQModel.load(str(output), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
        assert wrapper.model.config.llada_version == version
        state = wrapper.model.state_dict()
        assert state.keys() == source_state.keys()
        for name, value in source_state.items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0, msg=name)
        assert {
            name for name, mod in wrapper.model.named_modules() if isinstance(mod, AwqTorchLinear)
        } == expected_names
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(**ids).logits, expected, rtol=0, atol=0)
        for layer in wrapper.model.model.layers:
            if hasattr(layer.mlp, "gate"):
                assert layer.mlp.gate.expert_bias.dtype == torch.float32


def test_ambiguous_metadata_and_unsupported_overrides_fail_closed_then_retry(tmp_path):
    from gptqmodel import GPTQModel

    reference, marker = hostile_checkpoint(tmp_path)
    write_tokenizer(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    data.pop("llada_version")
    data["_name_or_path"] = "inclusionAI/LLaDA2.1-mini"
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Ambiguous.*llada2_moe"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported.*llada_version"):
        GPTQModel.load(str(tmp_path), llada_version="2.0-preview", device="cpu", dtype=torch.float32)
    loaded = GPTQModel.load(str(tmp_path), llada_version="2.1", device="cpu", dtype=torch.float32)
    ids = torch.tensor([[2, 3, 5, 6]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-5, atol=2e-7)
    assert not marker.exists()
    signature = tiny_config("2.2")
    signature.save_pretrained(tmp_path / "signature")
    with pytest.raises(ValueError, match="metadata requires"):
        LLaDA2MoeQModel.loader.from_config(
            LLaDA2MoeQModel.loader.load_config(tmp_path / "signature", llada_version="2.1")
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"llada_version": "future"}, "Unsupported"),
        ({"using_split_qkv_in_self_attention": True}, "packed"),
        ({"num_key_value_heads": 3}, "divisible"),
        ({"partial_rotary_factor": 0.25}, "rotary"),
        ({"n_group": 0}, "routing"),
        ({"n_group": 4}, "routing"),
        ({"topk_group": 0}, "routing"),
        ({"num_experts_per_tok": 5}, "num_experts_per_tok"),
    ],
)
def test_invalid_layouts_do_not_silently_construct(overrides, match):
    with pytest.raises(ValueError, match=match):
        LLaDA2MoeModelLM(tiny_config(**overrides))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"split_token_id": None}, "requires"),
        ({"split_token_id": 99}, "in-vocabulary"),
        ({"split_token_id": 33}, "distinct"),
        ({"expert_capacity": 1}, "capacity"),
        ({"block_size": 0}, "positive"),
        ({"block_size": 2.0}, "integers"),
    ],
)
def test_22_invalid_metadata_fails_closed(change, match):
    with pytest.raises(ValueError, match=match):
        LLaDA2MoeModelLM(tiny_config("2.2", **change))


def test_invalid_generation_retry_cache_rejection_and_22_batch_boundary():
    model = tiny_model()
    ids = torch.tensor([[2, 3]])
    for kwargs in (
        {"block_length": 0},
        {"steps": 0},
        {"gen_length": -1},
        {"mask_id": 99},
        {"eos_id": 99},
        {"minimal_topk": 0},
        {"use_cache": True},
    ):
        with pytest.raises(ValueError):
            model.generate(ids, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        model.generate(ids.expand(2, -1))
    with pytest.raises(ValueError, match="DynamicCache"):
        model(ids, past_key_values=())
    assert model.generate(ids, gen_length=0).shape == (1, 0)
    result = model.generate(
        ids, gen_length=2, block_length=2, steps=2, mask_id=35, eos_id=None, max_post_steps=0, threshold=-1, top_k=1
    )
    assert result.shape == (1, 2)
    model22 = tiny_model("2.2")
    with pytest.raises(ValueError, match="complete"):
        model22(torch.tensor([[2, 3, 4], [5, 6, 7]]))
    with pytest.raises(ValueError, match="divisible"):
        model22.generate(ids, gen_length=2, block_length=3, steps=2, mask_id=35, eos_id=None)


def test_21_nonconvergent_refinement_fails_with_explicit_budget(monkeypatch):
    model = tiny_model("2.1")

    def only_mask(input_ids, **kwargs):
        scores = torch.full((*input_ids.shape, 37), -50.0)
        scores[..., 35] = 50
        return SimpleNamespace(logits=scores)

    monkeypatch.setattr(model, "forward", only_mask)
    with pytest.raises(RuntimeError, match="max_steps_per_block"):
        model.generate(
            torch.tensor([[2]]), gen_length=1, block_length=2, steps=1, mask_id=35, eos_id=None, max_steps_per_block=3
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_default_lazy_offload_preserves_strict_fp32_nonzero_bias(tmp_path, dtype):
    """Shared LazyTurtle precision regression, separate from the decoder port."""
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import AWQConfig

    reference = tiny_model("2.0", use_bias=False)
    bias_name = "model.layers.1.mlp.gate.expert_bias"
    bias = reference.state_dict()[bias_name]
    bias.copy_(torch.tensor([0.001234567, -0.000987654, 0.001345678, -0.000456789]))
    reference.model.layers[1].mlp.gate.weight.data.normal_(std=0.5)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    write_tokenizer(tmp_path)
    wrapper = GPTQModel.load(
        str(tmp_path),
        dtype=dtype,
        backend=BACKEND.TORCH,
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
    )
    calibration = [
        {
            "input_ids": (torch.arange(8) + start).remainder(34).view(1, 8) + 2,
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }
        for start in [0, 8, 16, 24]
    ]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    torch.testing.assert_close(wrapper.model.state_dict()[bias_name], bias, rtol=0, atol=0)
