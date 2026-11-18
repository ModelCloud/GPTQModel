# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Instella qualification without executing checkpoint Python or downloading weights."""

import json

import pytest
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from gptqmodel.models.definitions.instella import InstellaQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.instella import InstellaConfig, InstellaForCausalLM
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


def tiny_config(**kwargs):
    options = {
        "vocab_size": 47, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 8,
        "pad_token_id": 1, "bos_token_id": 2, "eos_token_id": 3, "attention_dropout": 0.0,
    }
    options.update(kwargs)
    return InstellaConfig(**options)


def tiny_model(**kwargs):
    torch.manual_seed(71)
    model = InstellaForCausalLM(tiny_config(**kwargs)).eval()
    # Non-unit norm gains detect moving Q/K norms to the head dimension.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.linspace(0.6, 1.4, parameter.numel()).reshape_as(parameter))
    return model


def reference_logits(model, ids, mask=None, positions=None):
    """Independent transcription of AMD's eager equations, using only checkpoint tensors.

    Source: amd/Instella-3B-Instruct f83f548d8e20eccc8e403998970f42fb0acba657,
    modeling_instella.py:195-421, 634-727, 938-1004. No port helper is called.
    """
    cfg, state = model.config, model.state_dict()
    batch, length = ids.shape
    dim = cfg.hidden_size // cfg.num_attention_heads
    positions = torch.arange(length).unsqueeze(0) if positions is None else positions
    scaling = cfg.rope_scaling or {}
    kind = scaling.get("type", scaling.get("rope_type"))
    base = cfg.rope_theta
    seq_len = positions.max() + 1
    if kind == "dynamic" and seq_len > cfg.max_position_embeddings:
        base *= (scaling["factor"] * seq_len / cfg.max_position_embeddings - (scaling["factor"] - 1)) ** (
            dim / (dim - 2)
        )
    inv = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = positions.float() / scaling["factor"] if kind == "linear" else positions.float()
    angles = positions[..., None] * inv
    angles = torch.cat([angles, angles], -1)
    cos, sin = angles.cos().to(model.dtype)[:, None], angles.sin().to(model.dtype)[:, None]

    def norm(x, path):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
        return normalized.to(x.dtype) * state[path + ".weight"]

    def linear(x, path):
        return F.linear(x, state[path + ".weight"], state.get(path + ".bias"))

    def rotary(x):
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-second, first], -1) * sin

    hidden = F.embedding(ids, state["model.embed_tokens.weight"])
    causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    allowed = causal[None, None].expand(batch, 1, length, length)
    if mask is not None:
        allowed = allowed & mask[:, None, None, :].bool()
    for index in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{index}."
        x = norm(hidden, prefix + "pre_attention_layernorm")
        q = norm(linear(x, prefix + "self_attn.q_proj"), prefix + "self_attn.q_norm")
        k = norm(linear(x, prefix + "self_attn.k_proj"), prefix + "self_attn.k_norm")
        v = linear(x, prefix + "self_attn.v_proj")
        q = rotary(q.view(batch, length, cfg.num_attention_heads, dim).transpose(1, 2))
        k = rotary(k.view(batch, length, cfg.num_key_value_heads, dim).transpose(1, 2))
        v = v.view(batch, length, cfg.num_key_value_heads, dim).transpose(1, 2)
        k = k.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, 1)
        v = v.repeat_interleave(cfg.num_attention_heads // cfg.num_key_value_heads, 1)
        scores = (q @ k.transpose(-2, -1)) / dim**0.5
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = scores.float().softmax(-1).to(q.dtype)
        attention = (probs @ v).transpose(1, 2).reshape(batch, length, cfg.hidden_size)
        hidden = hidden + linear(attention, prefix + "self_attn.o_proj")
        x = norm(hidden, prefix + "pre_feedforward_layernorm")
        hidden = hidden + linear(
            F.silu(linear(x, prefix + "mlp.gate_proj")) * linear(x, prefix + "mlp.up_proj"),
            prefix + "mlp.down_proj",
        )
    return linear(norm(hidden, "model.norm"), "lm_head")


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_definition_owns_malicious_checkpoint_selection(tmp_path, monkeypatch, trust, bypass):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "evil.EvilConfig", "AutoModelForCausalLM": "evil.EvilModel",
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise AssertionError('checkpoint code executed')\n")
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    assert isinstance(InstellaQModel.loader, LocalModelLoader)
    assert resolve_trust_remote_code(str(tmp_path), trust_remote_code=trust) is False
    config = load_model_config(str(tmp_path), trust_remote_code=trust)
    assert type(config) is InstellaConfig
    loaded = InstellaQModel.loader.from_pretrained(tmp_path, config=config, trust_remote_code=trust)
    assert type(loaded) is InstellaForCausalLM
    ids = torch.tensor([[2, 6, 11, 9]])
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)
    fresh = InstellaQModel.loader.from_config(config, trust_remote_code=trust)
    assert type(fresh) is InstellaForCausalLM


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rope", [None, {"type": "linear", "factor": 2.0}, {"type": "dynamic", "factor": 2.0}])
def test_eager_matches_independent_amd_equations(heads, bias, rope):
    model = tiny_model(num_key_value_heads=heads, attention_bias=bias, rope_scaling=rope, attn_implementation="eager")
    assert model.config._attn_implementation == "eager"
    ids = torch.tensor([[2, 4, 8, 6, 12, 15, 9, 16, 21, 3], [2, 7, 13, 11, 5, 6, 8, 9, 17, 21]])
    mask = torch.tensor([[1] * 10, [1] * 8 + [0, 0]])
    expected = reference_logits(model, ids, mask)
    actual = model(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("rope", [None, {"type": "linear", "factor": 2.0}, {"type": "dynamic", "factor": 2.0}])
def test_cache_continuation_padding_causality_and_generation(dtype, backend, rope):
    model = tiny_model(rope_scaling=rope, attn_implementation=backend).to(dtype)
    assert model.config._attn_implementation == backend
    ids = torch.tensor([[2, 4, 8, 6, 12, 15], [1, 1, 2, 11, 5, 6]])
    mask = ids.ne(1).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        full = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        for cache in (DynamicCache(config=model.config), StaticCache(config=model.config, max_cache_len=12)):
            model(ids[:, :4], attention_mask=mask[:, :4], position_ids=positions[:, :4], past_key_values=cache)
            output = model(ids[:, 4:], attention_mask=mask, position_ids=positions[:, 4:], past_key_values=cache)
            assert output.past_key_values is cache
            assert cache.get_seq_length() == 6
            torch.testing.assert_close(output.logits, full[:, 4:], rtol=0.02, atol=0.004 if dtype != torch.float32 else 2e-6)
        changed = ids.clone()
        changed[:, -1] = 19
        perturb = model(changed, attention_mask=mask, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(perturb[:, 2:-1], full[:, 2:-1], rtol=0, atol=0)
        cached = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=True)
        uncached = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)


@pytest.mark.parametrize("rope", [None, {"type": "linear", "factor": 2.0}, {"type": "dynamic", "factor": 2.0}])
def test_safetensors_reload_rotary_and_retry(tmp_path, rope):
    model = tiny_model(rope_scaling=rope, rope_theta=20000.0)
    ids = torch.tensor([[2, 4, 8, 6]])
    expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("*.bin"))
    # Missing weights fail, then a valid retry on the same directory is usable.
    weights = tmp_path / "model.safetensors"
    weights.rename(tmp_path / "held.safetensors")
    with pytest.raises(OSError):
        InstellaForCausalLM.from_pretrained(tmp_path)
    (tmp_path / "held.safetensors").rename(weights)
    for attempt in range(2):
        restored = InstellaForCausalLM.from_pretrained(tmp_path)
        assert restored.config.rope_theta == 20000.0
        assert restored.model.layers[0].self_attn.rotary_emb.scaling_factor == (2.0 if rope else 1.0)
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        assert set(restored.state_dict()) == set(model.state_dict())
        for layer in restored.model.layers:
            inv = layer.self_attn.rotary_emb.inv_freq
            torch.testing.assert_close(inv, 1 / (20000 ** (torch.arange(0, 8, 2).float() / 8)))
        restored.save_pretrained(tmp_path)


def test_partial_decoder_cache_failure_requires_reset_before_retry(monkeypatch):
    model = tiny_model()
    ids = torch.tensor([[2, 4, 8, 6]])
    cache = DynamicCache(config=model.config)
    second = model.model.layers[1]
    original_forward = second.forward
    fired = []

    def fail_after_first_layer(*args, **kwargs):
        assert cache.layers[0].get_seq_length() == 4
        fired.append(True)
        raise RuntimeError("injected second decoder failure")

    monkeypatch.setattr(second, "forward", fail_after_first_layer)
    with pytest.raises(RuntimeError, match="injected second decoder"):
        model(ids, past_key_values=cache)
    assert fired == [True]
    assert cache.layers[0].get_seq_length() == 4
    assert cache.layers[1].get_seq_length() == 0
    cache.reset()
    monkeypatch.setattr(second, "forward", original_forward)
    retried = model(ids, past_key_values=cache).logits
    fresh = model(ids, past_key_values=DynamicCache(config=model.config)).logits
    torch.testing.assert_close(retried, fresh, rtol=0, atol=0)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_state_dict_assign_and_tied_reload(tmp_path, assign, tied):
    source = tiny_model(tie_word_embeddings=tied)
    target = tiny_model(tie_word_embeddings=tied)
    with torch.no_grad():
        target.model.layers[0].self_attn.q_norm.weight.fill_(3)
    target.load_state_dict(source.state_dict(), assign=assign, strict=True)
    ids = torch.tensor([[2, 4, 8, 6]])
    torch.testing.assert_close(target(ids).logits, source(ids).logits, rtol=0, atol=0)
    target.save_pretrained(tmp_path)
    restored = InstellaForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(ids).logits, source(ids).logits, rtol=0, atol=0)
    assert (restored.lm_head.weight is restored.model.embed_tokens.weight) == tied


def test_unsupported_flash_attention_fails_before_dispatch():
    with pytest.raises(ValueError, match="Flash Attention 2"):
        tiny_model(attn_implementation="flash_attention_2")
    assert tiny_model(attn_implementation="sdpa")(torch.tensor([[2, 4]])).logits.isfinite().all()


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_beam_generation_reorders_cache(backend):
    model = tiny_model(attn_implementation=backend)
    ids = torch.tensor([[2, 4, 8], [2, 9, 5]])
    options = {"num_beams": 3, "max_new_tokens": 4, "do_sample": False}
    assert torch.equal(
        model.generate(ids, use_cache=True, **options),
        model.generate(ids, use_cache=False, **options),
    )


def test_attention_outputs_loss_tuple_and_invalid_cache_retry():
    model = tiny_model()
    ids = torch.tensor([[2, 4, 8, 6]])
    with pytest.raises(TypeError, match="Transformers Cache"):
        model(ids, past_key_values=((torch.zeros(1), torch.zeros(1)),))
    output = model(ids, labels=ids, output_attentions=True, output_hidden_states=True, use_cache=False)
    assert torch.isfinite(output.loss)
    assert len(output.attentions) == 2
    assert len(output.hidden_states) == 3
    assert torch.count_nonzero(output.attentions[0].triu(1)) == 0
    assert isinstance(model(ids, return_dict=False, use_cache=False), tuple)
    torch.testing.assert_close(model(ids, num_logits_to_keep=1).logits, model(ids, logits_to_keep=1).logits)


def test_dynamic_rotary_cache_threshold_and_warm_buffer_contract(tmp_path):
    model = tiny_model(rope_scaling={"type": "dynamic", "factor": 2.0}, attn_implementation="eager")
    attention = model.model.layers[0].self_attn
    cache = DynamicCache(config=model.config)
    hidden = torch.randn(1, 2, 32)
    positions = torch.tensor([[0, 1]])
    attention(hidden, position_ids=positions, past_key_values=cache)
    original_keys = cache.layers[0].keys.clone()
    next_positions = torch.tensor([[8, 9]])
    attention(hidden, position_ids=next_positions, past_key_values=cache)
    # Instella retains already-rotated K: extending NTK does not re-rotate history.
    torch.testing.assert_close(cache.layers[0].keys[:, :, :2], original_keys, rtol=0, atol=0)
    base = 10000 * (2.0 * 10 / 8 - 1.0) ** (8 / 6)
    inv = 1 / (base ** (torch.arange(0, 8, 2).float() / 8))
    torch.testing.assert_close(attention.rotary_emb.inv_freq, inv)
    projected = attention.k_proj(hidden)
    normalized = projected.float() * torch.rsqrt(projected.float().square().mean(-1, keepdim=True) + 1e-5)
    normalized = normalized * attention.k_norm.weight
    keys = normalized.view(1, 2, 2, 8).transpose(1, 2)
    angles = next_positions[..., None] * inv
    angles = torch.cat([angles, angles], -1)[:, None]
    first, second = keys.chunk(2, -1)
    expected = keys * angles.cos() + torch.cat([-second, first], -1) * angles.sin()
    torch.testing.assert_close(cache.layers[0].keys[:, :, 2:], expected, rtol=2e-6, atol=2e-6)
    # AMD's warm dynamic buffer persists for a later short sequence; save/load
    # reconstructs the original non-persistent frequencies, rather than saving it.
    attention(hidden, position_ids=positions)
    torch.testing.assert_close(attention.rotary_emb.inv_freq, inv)
    model.save_pretrained(tmp_path)
    restored = InstellaForCausalLM.from_pretrained(tmp_path)
    torch.testing.assert_close(
        restored.model.layers[0].self_attn.rotary_emb.inv_freq,
        1 / (10000 ** (torch.arange(0, 8, 2).float() / 8)),
    )


def test_dynamic_rotary_uses_call_snapshot_after_buffer_replacement(monkeypatch):
    rotary = tiny_model(rope_scaling={"type": "dynamic", "factor": 2.0}).model.layers[0].self_attn.rotary_emb
    real_setattr = type(rotary).__setattr__
    replaced = []

    def publish_other_request(module, name, value):
        real_setattr(module, name, value)
        if module is rotary and name == "inv_freq":
            # Deterministic replacement at the publication boundary models a
            # second request publishing different NTK frequencies before use.
            replaced.append(True)
            real_setattr(module, name, value * 3)

    monkeypatch.setattr(type(rotary), "__setattr__", publish_other_request)
    positions = torch.tensor([[8, 9]])
    cos, sin = rotary(torch.zeros(1, 2, 8), positions)
    assert replaced == [True]
    base = 10000 * (2.0 * 10 / 8 - 1.0) ** (8 / 6)
    inv = 1 / (base ** (torch.arange(0, 8, 2).float() / 8))
    angles = positions[..., None] * inv
    angles = torch.cat([angles, angles], -1)
    # The scalar oracle evaluates NTK powers in Python float; the runtime uses
    # FP32 tensors. Bound that rounding well below the injected 3x corruption.
    torch.testing.assert_close(cos, angles.cos(), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(sin, angles.sin(), rtol=2e-6, atol=2e-7)


def test_invalid_scaling_rejected_then_valid_config():
    with pytest.raises(ValueError, match="rope_scaling"):
        tiny_config(rope_scaling={"type": "unknown", "factor": 2.0})
    with pytest.raises(ValueError, match="divide"):
        tiny_config(num_key_value_heads=3)
    assert tiny_model(rope_scaling={"rope_type": "linear", "factor": 2.0})(torch.tensor([[2, 4]])).logits.isfinite().all()


@pytest.mark.parametrize("fallback", [None, {}])
def test_awq_scale_search_quantizes_and_packs_every_linear(monkeypatch, fallback):
    from types import SimpleNamespace

    from torch import nn

    from gptqmodel.looper import awq_processor
    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.loop_processor import CalibrationConfig
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = tiny_model(attention_bias=True).half()
    wrapper = InstellaQModel(model=model, quantized=False, quantize_config=None)
    layer = model.model.layers[0]
    qcfg = AWQConfig(device="cpu", bits=4, group_size=16, fallback=fallback)
    processor = AWQProcessor(
        tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(),
        gptq_model=SimpleNamespace(
            model=model, qlinear_kernel=AwqTorchLinear, rotary_embedding=None,
            lm_head="lm_head", awq_get_modules_for_scaling=wrapper.awq_get_modules_for_scaling,
        ), model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args: None
    processor.log_new_row = lambda *args: None
    state = processor._get_layer_state(0)
    state.layer_module = layer
    state.modules = {
        name: NamedModule(module, name, "model.layers.0." + name, 0)
        for name, module in layer.named_modules() if isinstance(module, nn.Linear)
    }
    assert len(state.modules) == 7
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    features = {name: torch.randn(5, named.module.in_features).half() for name, named in state.modules.items()}
    monkeypatch.setattr(processor, "_layer_input_features", lambda _: features)
    norms = {name: tensor.clone() for name, tensor in layer.state_dict().items() if "norm" in name}
    original = {name: named.module.weight.clone() for name, named in state.modules.items()}
    applied = []
    original_apply_scale = awq_processor.apply_scale

    def record_apply_scale(module, scales_list, *args, **kwargs):
        applied.extend((source, tuple(targets)) for source, targets, _scales, _loss in scales_list)
        return original_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", record_apply_scale)
    processor._quantize_layer(0, state)
    assert state.quantized
    # GQA leaves v_proj -> o_proj width-ineligible; the other folds are searched.
    assert applied == [
        ("pre_attention_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("pre_feedforward_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ("mlp.up_proj", ("mlp.down_proj",)),
    ]
    # Decoder norms absorb the scales; the full-width Q/K norms read the
    # unchanged q/k projections and must stay untouched.
    for name, weight in norms.items():
        assert torch.equal(layer.state_dict()[name], weight) is ("self_attn." in name), name
    ids = torch.tensor([[2, 4, 8, 6]])
    before_pack = model(ids, use_cache=False).logits
    for name, named in state.modules.items():
        assert not torch.equal(original[name], named.module.weight)
        assert {"q_scales", "q_zeros", "wq"} <= set(named.state)
        dense = named.module
        packed = AwqTorchLinear(
            bits=4, group_size=16, sym=qcfg.sym, desc_act=False,
            in_features=dense.in_features, out_features=dense.out_features,
            bias=dense.bias is not None, dtype=torch.float16,
        )
        packed.pack(dense, named.state["q_scales"], named.state["q_zeros"])
        assert packed.qweight.dtype == torch.int32
        assert packed.qweight.numel() > 0
        parent_name, _, leaf = name.rpartition(".")
        setattr(layer.get_submodule(parent_name), leaf, packed)
    actual = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, before_pack, rtol=0.02, atol=0.003)


@pytest.mark.parametrize("trust", [False, True])
def test_public_dense_load_uses_packaged_model(tmp_path, monkeypatch, trust):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel

    tiny_model(rope_scaling={"rope_type": "linear", "factor": 2.0}).save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(47)}, unk_token="0")),
        unk_token="0", pad_token="1", bos_token="2", eos_token="3",
    )
    tokenizer.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=None, device="cpu", trust_remote_code=trust)
    assert isinstance(wrapper, InstellaQModel)
    assert type(wrapper.model) is InstellaForCausalLM
    assert wrapper.trust_remote_code is False
    assert wrapper.model.config.rope_scaling["factor"] == 2.0
    assert wrapper.model(torch.tensor([[2, 4, 8]])).logits.isfinite().all()


@pytest.mark.parametrize("legacy", [None, {"type": "linear", "factor": 2.0}, {"rope_type": "dynamic", "factor": 3}])
def test_rope_is_stored_as_tf5_rope_parameters_without_deprecated_alias(tmp_path, legacy):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = tiny_config(rope_scaling=legacy, rope_theta=20000.0)
        config.save_pretrained(tmp_path)
        payload = json.loads((tmp_path / "config.json").read_text())
        restored = InstellaConfig.from_pretrained(tmp_path)
    expected = {"rope_type": "default", "rope_theta": 20000.0}
    if legacy:
        expected = {"rope_type": legacy.get("type", legacy.get("rope_type")), "rope_theta": 20000.0,
                    "factor": legacy["factor"]}
    assert payload["rope_parameters"] == expected and "rope_scaling" not in payload
    assert restored.rope_parameters == expected and restored.rope_theta == 20000.0


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_gradient_checkpointing_matches_plain_backward(backend):
    model = tiny_model(attn_implementation=backend).train()
    ids = torch.tensor([[2, 4, 8, 6]])
    model(ids, labels=ids, use_cache=False).loss.backward()
    reference = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)
    model.activation_checkpointing_enable()
    model(ids, labels=ids, use_cache=False).loss.backward()
    grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    assert grads.keys() == reference.keys()
    for name, grad in reference.items():
        torch.testing.assert_close(grads[name], grad, rtol=0, atol=0)
