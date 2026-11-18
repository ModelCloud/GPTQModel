# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import re

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from gptqmodel.models.definitions.solar_open2 import SolarOpen2QModel
from transformers.cache_utils import Cache

from gptqmodel.models.ports.solar_open2 import SolarOpen2Config, SolarOpen2DynamicCache, SolarOpen2ForCausalLM
from gptqmodel.utils.hf import build_shell_model, load_model_config


def tiny_values(**kwargs):
    values = {
        "model_type": "solar_open2",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "gqa_layers": [0],
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(kwargs)
    return values


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "1")])
def test_package_selection_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    values = tiny_values(
        auto_map={
            "AutoConfig": "hostile.EvilConfig",
            "AutoModelForCausalLM": "hostile.EvilModel",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(values))
    (tmp_path / "hostile.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    config = load_model_config(tmp_path, trust_remote_code=trust)
    shell = build_shell_model(
        SolarOpen2QModel.loader,
        config=config,
        trust_remote_code=trust,
        device_map={"": "cpu"},
        _fast_init=True,
    )
    model = SolarOpen2QModel.loader.from_config(config, trust_remote_code=trust)
    assert type(config).__module__.startswith("gptqmodel.models.ports.solar_open2.")
    assert type(model).__module__.startswith("gptqmodel.models.ports.solar_open2.")
    assert type(shell) is type(model)
    with torch.inference_mode():
        assert model(torch.tensor([[1, 7, 9]]), use_cache=False).logits.shape == (1, 3, 64)


def tiny_model(**kwargs):
    torch.manual_seed(271)
    return SolarOpen2ForCausalLM(SolarOpen2Config(**tiny_values(**kwargs))).eval()


def write_tokenizer(path):
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{f"w{i}": i for i in range(4, 64)}}, unk_token="<unk>"
        )
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    ).save_pretrained(path)


def linear(x, module):
    return F.linear(x, module.weight.double(), None if module.bias is None else module.bias.double())


def norm(x, module):
    return x * (x.square().mean(-1, keepdim=True) + module.variance_epsilon).rsqrt() * module.weight.double()


def mlp_oracle(x, module):
    return linear(F.silu(linear(x, module.gate_proj)) * linear(x, module.up_proj), module.down_proj)


def moe_oracle(x, module):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    gate = module.gate
    scores = F.linear(x, gate.weight.double()).sigmoid()
    choice = scores + gate.e_score_correction_bias.double()
    grouped = choice.reshape(-1, gate.num_group, gate.num_experts // gate.num_group)
    selected_groups = grouped.sort(-1, descending=True).values[..., :2].sum(-1).topk(gate.topk_group, -1).indices
    allowed = F.one_hot(selected_groups, gate.num_group).sum(1).bool()
    choice = grouped.masked_fill(~allowed[..., None], -torch.inf).reshape_as(choice)
    indices = choice.topk(gate.top_k, -1, sorted=False).indices
    weights = scores.gather(1, indices)
    if gate.norm_topk_prob:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    out = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(gate.top_k):
            out[token] += (
                mlp_oracle(x[token], module.experts[indices[token, slot]])
                * weights[token, slot]
                * gate.routed_scaling_factor
            )
    if module.shared_experts is not None:
        out += mlp_oracle(x, module.shared_experts)
    return out.reshape(shape)


def attention_oracle(x, module, positions, padding):
    batch, seq, _ = x.shape
    cfg = module.config
    q = linear(x, module.q_proj).reshape(batch, seq, cfg.num_attention_heads, cfg.head_dim)
    k = linear(x, module.k_proj).reshape(batch, seq, cfg.num_key_value_heads, cfg.head_dim)
    v = linear(x, module.v_proj).reshape_as(k)
    if cfg.use_qk_norm:
        q, k = norm(q, module.q_norm), norm(k, module.k_norm)
    if cfg.use_rope:
        dim = int(cfg.head_dim * cfg.rope_parameters.get("partial_rotary_factor", 1.0))
        inv = cfg.rope_parameters["rope_theta"] ** (-torch.arange(0, dim, 2).double() / dim)
        angles = (positions[..., None].float() * inv.float()).double()
        angles = torch.cat((angles, angles), -1)[:, :, None]

        def rotate(t):
            first, second = t[..., :dim].chunk(2, -1)
            return torch.cat(
                (t[..., :dim] * angles.cos() + torch.cat((-second, first), -1) * angles.sin(), t[..., dim:]), -1
            )

        q, k = rotate(q), rotate(k)
    k = k.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, 2)
    v = v.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, 2)
    scores = torch.einsum("bthd,bshd->bhts", q, k) / cfg.head_dim**0.5
    allowed = torch.ones(seq, seq, dtype=torch.bool).tril()[None, None] & padding[:, None, None].bool()
    probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    result = torch.einsum("bhts,bshd->bthd", probs, v).reshape(batch, seq, -1)
    if cfg.use_gqa_gate:
        result *= linear(x, module.g_proj).sigmoid()
    return linear(result, module.o_proj)


def kda_oracle(x, module, padding):
    x = x * padding[..., None]
    batch, seq, _ = x.shape
    tensors = []
    for projection, convolution, heads in [
        (module.q_proj, module.q_conv1d, module.num_heads),
        (module.k_proj, module.k_conv1d, module.num_kv_heads),
        (module.v_proj, module.v_conv1d, module.num_kv_heads),
    ]:
        raw = linear(x, projection)
        convolved = torch.zeros_like(raw)
        for t in range(seq):
            for lag in range(min(t + 1, module.conv_size)):
                convolved[:, t] += raw[:, t - lag] * convolution.weight[:, 0, module.conv_size - 1 - lag].double()
        tensors.append(F.silu(convolved).reshape(batch, seq, heads, module.head_dim))
    q, k, v = tensors
    k = k.repeat_interleave(module.n_rep, 2)
    v = v.repeat_interleave(module.n_rep, 2)
    q /= (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k /= (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    raw_gate = (
        linear(x, module.f_proj) if module.use_full_proj else linear(linear(x, module.f_a_proj), module.f_b_proj)
    )
    raw_gate = raw_gate.reshape_as(q) + module.dt_bias.double().reshape(module.num_heads, module.head_dim)
    decay = -F.softplus(raw_gate) * module.A_log.double().exp()
    if module.gate_lower_bound is not None:
        decay = decay.clamp_min(module.gate_lower_bound)
    beta = linear(x, module.b_proj).sigmoid() * (2 if module.allow_neg_eigval else 1)
    state = torch.zeros(batch, module.num_heads, module.head_dim, module.head_dim, dtype=torch.float64)
    result = []
    for t in range(seq):
        state = state * decay[:, t, :, :, None].exp()
        prediction = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
        update = beta[:, t, :, None] * (v[:, t] - prediction)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], update)
        result.append(torch.einsum("bhk,bhkv->bhv", q[:, t] / module.head_dim**0.5, state))
    result = torch.stack(result, 1)
    output_gate = (
        linear(x, module.g_proj) if module.use_full_proj else linear(linear(x, module.g_a_proj), module.g_b_proj)
    )
    result = norm(result, module.o_norm) * output_gate.reshape_as(result).sigmoid()
    return linear(result.reshape(batch, seq, -1), module.o_proj), state


@pytest.mark.parametrize("kv_heads,full_proj,bias", [(1, False, False), (2, True, True), (4, False, True), (4, True, False)])
def test_independent_float64_full_decoder(kv_heads, full_proj, bias):
    model = tiny_model(
        num_key_value_heads=kv_heads,
        use_qk_norm=bias,
        use_rope=bias,
        attention_bias=bias,
        use_gqa_gate_bias=bias,
        kda_use_full_proj=full_proj,
        linear_attn_config={"head_dim": 8, "num_heads": 4, "num_kv_heads": kv_heads, "short_conv_kernel_size": 3},
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.3,
        norm_topk_prob=not bias,
    ).double()
    gen = torch.Generator().manual_seed(114)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm.weight" not in name:
                parameter.copy_(torch.randn(parameter.shape, generator=gen).double() * 0.21)
        for layer in model.model.layers:
            layer.mlp.gate.e_score_correction_bias.copy_(torch.tensor([0.04, -0.1, 0.08, -0.02]))
        ids = torch.tensor([[1, 5, 7, 8, 9], [1, 6, 4, 3, 10]])
        padding = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        positions = torch.arange(65531, 65536)[None]
        expected = model.model.embed_tokens(ids).double()
        for layer in model.model.layers:
            normalized = norm(expected, layer.input_layernorm)
            if layer.is_linear_attn:
                contribution, _ = kda_oracle(normalized, layer.self_attn, padding)
            else:
                contribution = attention_oracle(normalized, layer.self_attn, positions, padding)
            expected += contribution
            expected += moe_oracle(norm(expected, layer.post_attention_layernorm), layer.mlp)
        expected = linear(norm(expected, model.model.norm), model.lm_head)
        actual = model(ids, attention_mask=padding, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


LAYOUTS = [
    ["full_attention", "linear_attention"],
    ["linear_attention", "full_attention"],
    ["linear_attention", "linear_attention"],
    ["full_attention", "full_attention"],
]


@pytest.mark.parametrize(
    "dtype,layout,implementation",
    [(torch.float32, layout, implementation) for layout in LAYOUTS for implementation in ("eager", "sdpa")]
    + [(torch.float16, LAYOUTS[0], "sdpa"), (torch.bfloat16, LAYOUTS[1], "eager")],
)
def test_cache_continuation_generation_and_causality(dtype, layout, implementation):
    model = tiny_model(
        layer_types=layout, kda_use_full_proj=True, use_rope=True, use_qk_norm=True, first_k_dense_replace=1
    ).to(dtype)
    model.set_attn_implementation(implementation)
    ids = torch.tensor([[1, 5, 7, 8, 9, 10], [1, 6, 4, 3, 10, 11]])
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        part = model(ids[:, :1], use_cache=True)
        assert isinstance(part.past_key_values, SolarOpen2DynamicCache)
        assert isinstance(part.past_key_values, Cache)
        part = model(ids[:, 1:5], past_key_values=part.past_key_values)
        part = model(ids[:, 5:], past_key_values=part.past_key_values)
        assert part.past_key_values.get_seq_length() == ids.shape[1]
        assert all(part.past_key_values.has_previous_state(i) for i, kind in enumerate(layout) if kind != "full_attention")
        tolerance = 1e-6 if dtype == torch.float32 else 2e-3
        torch.testing.assert_close(part.logits, expected[:, -1:], atol=tolerance, rtol=tolerance)
        changed = ids.clone()
        changed[:, -1] = 17
        torch.testing.assert_close(model(changed, use_cache=False).logits[:, :-1], expected[:, :-1], rtol=0, atol=0)
        greedy = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
        replay = ids[:, :3].clone()
        for _ in range(3):
            next_token = model(replay, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            replay = torch.cat((replay, next_token), -1)
        assert torch.equal(greedy, replay)
        beams_cached = model.generate(ids[:, :3], max_new_tokens=3, num_beams=3, do_sample=False, eos_token_id=None)
        beams_uncached = model.generate(
            ids[:, :3], max_new_tokens=3, num_beams=3, do_sample=False, eos_token_id=None, use_cache=False
        )
        assert torch.equal(beams_cached, beams_uncached)
        part.past_key_values.reset()
        assert part.past_key_values.get_seq_length() == 0
        assert not any(
            part.past_key_values.has_previous_state(i) for i, kind in enumerate(layout) if kind != "full_attention"
        )
        restarted = model(ids, past_key_values=part.past_key_values).logits
        torch.testing.assert_close(restarted, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype,assign", [(torch.float32, False), (torch.float16, True), (torch.bfloat16, False)])
def test_dense_safetensors_two_cycles_assignment_and_repair(tmp_path, dtype, assign):
    source = tiny_model().to(dtype)
    ids = torch.tensor([[1, 7, 9, 5]])
    with torch.inference_mode():
        expected = source(ids, use_cache=False).logits
    clone = tiny_model().to(dtype)
    clone.load_state_dict(source.state_dict(), assign=assign)
    for cycle in range(2):
        path = tmp_path / str(cycle)
        clone.save_pretrained(path, safe_serialization=True)
        tensors = path / "model.safetensors"
        original = tensors.read_bytes()
        tensors.write_bytes(b"broken tensor header")
        with pytest.raises(Exception, match="[Hh]eader|safetensors|deserializ"):
            SolarOpen2QModel.loader.from_pretrained(path, dtype=dtype)
        tensors.write_bytes(original)
        clone = SolarOpen2QModel.loader.from_pretrained(path, dtype=dtype).eval()
        assert clone.model.layers[0].mlp.gate.e_score_correction_bias.dtype == torch.float32
        assert set(clone.state_dict()) == set(source.state_dict())
        for key, value in source.state_dict().items():
            assert torch.equal(clone.state_dict()[key], value), key
        with torch.inference_mode():
            torch.testing.assert_close(clone(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        assert not list(path.glob("*.bin"))


@pytest.mark.parametrize("trust,bypass", [(False, None), (True, "1")])
def test_public_dense_hostile_checkpoint_policy_two_reloads(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import BACKEND, GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    path = tmp_path / "source"
    source = tiny_model()
    source.generation_config.eos_token_id = [2, 9]
    source.generation_config.do_sample = True
    source.save_pretrained(path)
    write_tokenizer(path)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "hostile.EvilConfig", "AutoModelForCausalLM": "hostile.EvilModel"}
    (path / "config.json").write_text(json.dumps(payload))
    (path / "hostile.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    (path / "custom_generate").mkdir()
    (path / "custom_generate/generate.py").write_text("raise AssertionError('checkpoint generation executed')\n")
    ids = torch.tensor([[1, 7, 9, 5]])
    with torch.no_grad():
        expected = source(ids, use_cache=False).logits
    wrapper = GPTQModel.load(str(path), backend=BACKEND.TORCH, dtype=torch.float32, trust_remote_code=trust)
    for cycle in range(3):
        assert type(wrapper) is SolarOpen2QModel
        assert type(wrapper.model) is SolarOpen2ForCausalLM
        assert wrapper.model.generation_config.eos_token_id == [2, 9]
        assert wrapper.model.generation_config.do_sample is True
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids, use_cache=False).logits, expected, rtol=0, atol=0)
            assert wrapper.generate(ids, max_new_tokens=2, do_sample=False).shape[1] > ids.shape[1]
        if cycle < 2:
            target = tmp_path / f"saved-{cycle}"
            wrapper.model.save_pretrained(target, safe_serialization=True)
            wrapper.tokenizer.save_pretrained(target)
            wrapper = GPTQModel.load(str(target), backend=BACKEND.TORCH, dtype=torch.float32, trust_remote_code=trust)


@pytest.mark.parametrize("offload,full_proj,dense_first", [(False, False, False), (True, True, False), (False, False, True)])
def test_public_awq_all_projections_two_exact_reloads(tmp_path, offload, full_proj, dense_first):
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    path = tmp_path / "source"
    source = tiny_model(kda_use_full_proj=full_proj, first_k_dense_replace=int(dense_first))
    for layer in source.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.tensor([0.000300013, -0.000500017, 0.000700019, -0.000900023])
            )
    source.save_pretrained(path)
    write_tokenizer(path)
    original = {k: v.clone() for k, v in source.state_dict().items()}
    qcfg = AWQConfig(
        bits=4,
        group_size=16,
        device="cpu",
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapper = GPTQModel.load(str(path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    samples = [
        {
            "input_ids": torch.tensor([[1, 7, 9, 5, 12, 17, 21, 32, 15, 24, 36, 42, 51, 61, 11, 8]]),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
    ]
    wrapper.quantize(CalibrationConfig(dataset=samples, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = set()
    for index, layer in enumerate(source.model.layers):
        for name, module in layer.named_modules():
            spec = wrapper.module_plan().find(name)
            if isinstance(module, torch.nn.Linear) and spec is not None and spec.quantize:
                expected.add(f"model.layers.{index}.{name}")
    assert packed == expected
    assert len(packed) == (27 if dense_first else 40 if full_proj else 39)
    searched = {(int(row["layer"]), str(row["module"])) for row in wrapper.quant_log if row["loss"] != "not applicable"}
    assert {(0, "self_attn.q_proj"), (1, "self_attn.q_proj"), (1, "mlp.shared_experts.gate_proj")} <= searched
    # AWQ folds into both decoder norms and compensates the dense consumers
    # that read the same normed input (router and KDA decay/beta/gate inputs).
    folded = ("input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight",
              "self_attn.b_proj.weight", "self_attn.f_proj.weight", "self_attn.f_a_proj.weight",
              "self_attn.g_a_proj.weight")
    before = {k: v.detach().clone() for k, v in wrapper.model.state_dict().items()}
    for key, value in before.items():
        if key in original and key.endswith(folded):
            assert not torch.equal(value, original[key].to(value.dtype)), key
        elif key in original:
            if key.endswith("e_score_correction_bias"):
                assert value.dtype == torch.float32
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, original[key].to(value.dtype), rtol=0, atol=0)
    ids = samples[0]["input_ids"]
    with torch.no_grad():
        logits = wrapper.model(ids, use_cache=False).logits
        assert torch.isfinite(logits).all()
    for cycle in range(2):
        target = tmp_path / f"quantized-{cycle}"
        wrapper.save(str(target))
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        wrapper = GPTQModel.load(str(target), backend=BACKEND.TORCH, dtype=torch.float16, device="cpu")
        after = wrapper.model.state_dict()
        assert set(after) == set(before)
        for key in before:
            assert torch.equal(before[key], after[key]), key
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids, use_cache=False).logits, logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    "values,match",
    [
        ({"layer_types": ["full_attention"]}, "layer_types"),
        ({"layer_types": ["foreign", "linear_attention"]}, "layer_types"),
        ({"num_key_value_heads": 3}, "divisible"),
        ({"n_routed_experts": 3, "n_group": 2}, "groups"),
        ({"topk_group": 2}, "topk_group"),
        ({"n_group": 2, "topk_group": 1, "num_experts_per_tok": 3}, "supply"),
    ],
)
def test_invalid_layout_fails_then_valid_retry(values, match):
    with pytest.raises(ValueError, match=match):
        tiny_model(**values)
    assert tiny_model().config.model_type == "solar_open2"


def test_config_input_immutability_and_optional_shared_experts():
    original = {"head_dim": 8, "num_heads": 4, "num_kv_heads": 2, "short_conv_kernel_size": 2}
    snapshot = copy.deepcopy(original)
    model = tiny_model(linear_attn_config=original, n_shared_experts=0)
    model.config.linear_attn_config["short_conv_kernel_size"] = 17
    assert original == snapshot
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[1, 7, 8]]), use_cache=False).logits).all()


@pytest.mark.parametrize("full_proj", [False, True])
def test_independent_kda_final_state_and_interleaved_cache_ownership(full_proj):
    model = tiny_model(
        layer_types=["linear_attention"] * 2,
        kda_use_full_proj=full_proj,
        kda_gate_lower_bound=-0.5,
        kda_allow_neg_eigval=False,
    )
    attention = model.model.layers[0].self_attn
    inputs = torch.randn(2, 6, 32) * 1.7
    padding = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        expected, state = kda_oracle(inputs.double(), attention, padding)
        cache = SolarOpen2DynamicCache(model.config)
        actual, _ = attention(inputs, attention_mask=padding, past_key_values=cache)
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(cache.layers[0].recurrent_states[0].double(), state, atol=1e-6, rtol=1e-5)
        ids = torch.tensor([[1, 7, 8, 9, 10, 11], [1, 12, 6, 4, 7, 8]])
        first = model(ids[:, :2]).past_key_values
        other = model(ids.flip(-1)[:, :4]).past_key_values
        assert first is not other
        before = other.layers[0].recurrent_states[0].clone()
        continued = model(ids[:, 2:], past_key_values=first).logits
        torch.testing.assert_close(continued, model(ids, use_cache=False).logits[:, 2:], atol=1e-6, rtol=1e-5)
        assert torch.equal(other.layers[0].recurrent_states[0], before)
        with pytest.raises(RuntimeError, match="past states"):
            first.crop(1)
        fresh = model(ids).past_key_values
        assert all(fresh.has_previous_state(i) for i in range(2))
        assert fresh.get_seq_length() == ids.shape[1]


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_left_padded_prefill_and_continuation(implementation, layout):
    model = tiny_model(layer_types=layout, use_rope=True, use_gqa_gate=False)
    model.set_attn_implementation(implementation)
    assert model.config._attn_implementation == implementation
    ids = torch.tensor([[0, 0, 1, 7, 8, 9], [1, 5, 6, 7, 8, 9]])
    mask = ids.ne(0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        part = model(ids[:, :5], attention_mask=mask[:, :5], position_ids=positions[:, :5])
        part = model(
            ids[:, 5:], attention_mask=mask, position_ids=positions[:, 5:], past_key_values=part.past_key_values
        )
        torch.testing.assert_close(part.logits, expected[:, -1:], atol=1e-6, rtol=1e-5)
        unpadded = model(ids[:1, 2:], use_cache=False).logits
        torch.testing.assert_close(expected[:1, -1], unpadded[:, -1], atol=1e-6, rtol=1e-5)


def test_tuple_hidden_states_attentions_and_partial_rope():
    model = tiny_model(
        use_rope=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        use_qk_norm=True,
    )
    model.set_attn_implementation("eager")
    ids = torch.tensor([[1, 7, 9, 5]])
    with torch.no_grad():
        output = model(ids, output_hidden_states=True, output_attentions=True, use_cache=False)
        as_tuple = model(ids, return_dict=False, use_cache=False)
        assert len(output.hidden_states) == model.config.num_hidden_layers + 1
        assert len(output.attentions) == 1
        assert output.attentions[0].shape == (1, 4, 4, 4)
        torch.testing.assert_close(output.logits, as_tuple[0], atol=0, rtol=0)
        embed = norm(model.model.embed_tokens(ids).double(), model.model.layers[0].input_layernorm)
        positions = torch.arange(4)[None]
        cos, sin = model.model.rotary_emb(embed, positions)
        actual, _ = model.model.layers[0].self_attn(
            embed.float(),
            (cos.float(), sin.float()),
            attention_mask=torch.triu(torch.full((4, 4), -torch.inf), 1)[None, None],
        )
        expected = attention_oracle(embed, model.model.layers[0].self_attn, positions, torch.ones_like(ids))
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-5)


def test_parallel_plan_targets_per_expert_layout_and_legacy_rope_fields():
    model = tiny_model(first_k_dense_replace=1)
    names = {name for name, _ in model.model.named_modules()} | {name for name, _ in model.model.named_parameters()}
    for pattern in model.config.base_model_tp_plan:
        regex = re.compile("^" + re.escape(pattern).replace(r"\*", r"\d+") + "$")
        assert any(regex.match(name) for name in names), pattern
    assert model.config.base_model_ep_plan is None  # grouped-GEMM EP needs fused experts
    assert SolarOpen2Config(**tiny_values()).rope_parameters == {
        "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
    }
    legacy = SolarOpen2Config(**tiny_values(rope_theta=5e5, rope_scaling={"rope_type": "linear", "factor": 2.0}))
    assert legacy.rope_parameters == {
        "rope_type": "linear", "factor": 2.0, "rope_theta": 5e5, "partial_rotary_factor": 1.0,
    }
