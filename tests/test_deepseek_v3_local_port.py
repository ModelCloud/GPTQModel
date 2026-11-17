# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native DeepSeek-V3 and standalone Kimi-K2 checkpoint loading contracts."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_payload(model_type="deepseek_v3", q_lora_rank=8, **overrides):
    payload = {
        "model_type": model_type, "architectures": ["DeepseekV3ForCausalLM"],
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "n_group": 1, "topk_group": 1, "first_k_dense_replace": 1, "moe_layer_freq": 1,
        "kv_lora_rank": 8, "q_lora_rank": q_lora_rank, "qk_nope_head_dim": 4, "qk_rope_head_dim": 4,
        "v_head_dim": 8, "max_position_embeddings": 64, "num_nextn_predict_layers": 0,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
        "routed_scaling_factor": 2.827 if model_type == "kimi_k2" else 2.5,
        "rope_theta": 50000.0 if model_type == "kimi_k2" else 10000.0,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    }
    payload.update(overrides)
    return payload


def poison_checkpoint(path, payload):
    path.mkdir(exist_ok=True)
    marker = path / "EXECUTED"
    payload = copy.deepcopy(payload)
    payload["auto_map"] = {
        "AutoConfig": "checkpoint_python.RemoteConfig",
        "AutoModelForCausalLM": "checkpoint_python.RemoteModel",
    }
    (path / "checkpoint_python.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('remote code executed')\n"
        "raise AssertionError('checkpoint Python must not run')\n"
    )
    (path / "config.json").write_text(json.dumps(payload))
    return marker


def checkpoint_state_dict(model):
    """Published checkpoints own separate per-expert gate/up/down weights."""
    state = {}
    for key, value in model.state_dict().items():
        if key.endswith("experts.gate_up_proj"):
            prefix = key.removesuffix("gate_up_proj")
            for i, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                state[f"{prefix}{i}.gate_proj.weight"] = gate.clone()
                state[f"{prefix}{i}.up_proj.weight"] = up.clone()
        elif key.endswith("experts.down_proj"):
            prefix = key.removesuffix("down_proj")
            for i, expert in enumerate(value):
                state[f"{prefix}{i}.down_proj.weight"] = expert.clone()
        else:
            state[key] = value.clone()
    return state


def write_checkpoint(path, payload):
    # The reference uses the native constructor directly, independently of the loader under test.
    config = DeepseekV3Config.from_dict(dict(payload, model_type="deepseek_v3"))
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = DeepseekV3ForCausalLM(config).eval()
        with torch.no_grad():
            reference.model.layers[1].mlp.gate.e_score_correction_bias.copy_(torch.tensor([.1, -.2, .3, -.4]))
    reference.save_pretrained(path, safe_serialization=True)
    save_file(checkpoint_state_dict(reference), str(path / "model.safetensors"), metadata={"format": "pt"})
    marker = poison_checkpoint(path, payload)
    return reference, marker


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_definition_selects_native_config_and_model_despite_auto_map(tmp_path, monkeypatch, model_type, trust, bypass):
    marker = poison_checkpoint(tmp_path, tiny_payload(model_type))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass is explicitly under test.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert isinstance(config, DeepseekV3Config)
    assert config.model_type == "deepseek_v3", "Kimi-K2 must use native expert checkpoint conversion, not an unknown alias"
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is DeepSeekV3QModel
    assert not DeepSeekV3QModel.require_trust_remote_code
    with torch.device("meta"):
        model = DeepSeekV3QModel.loader.from_config(config, trust_remote_code=trust, code_revision="untrusted")
    assert isinstance(model, DeepseekV3ForCausalLM)
    assert type(model).forward is DeepseekV3ForCausalLM.forward
    assert next(model.parameters()).device.type == "meta"
    assert not marker.exists()


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_native_checkpoint_cache_generation_and_safetensors_roundtrip(tmp_path, model_type, q_lora_rank):
    payload = tiny_payload(model_type, q_lora_rank, n_shared_experts=2)
    reference, marker = write_checkpoint(tmp_path, payload)
    config = load_model_config(tmp_path)
    config._attn_implementation = "eager"
    model, info = DeepSeekV3QModel.loader.from_pretrained(
        tmp_path, config=config, output_loading_info=True, trust_remote_code=True,
    )
    model.eval()
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert set(model.state_dict()) == set(reference.state_dict())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[key], rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 5, 7, 3], [1, 8, 4, 6, 9]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
        prefill = model(ids[:, :3], use_cache=True)
        assert prefill.past_key_values.get_seq_length() == 3
        continued = model(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)
        assert continued.past_key_values.get_seq_length() == 5
        torch.testing.assert_close(continued.logits, expected[:, 3:], rtol=1e-5, atol=1e-6)
        generated = model.generate(ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(generated, reference.generate(
            ids[:, :3], max_new_tokens=3, do_sample=False, eos_token_id=None,
        ))
        altered = ids.clone()
        altered[:, -1] = 12
        torch.testing.assert_close(model(altered).logits[:, :-1], expected[:, :-1], rtol=1e-5, atol=1e-6)
    # Normal production conversion must expose all original checkpoint expert owners for both aliases.
    assert _convert_model_with_defuser(DeepSeekV3QModel, model, cleanup_original=True)
    assert set(model.state_dict()) == set(checkpoint_state_dict(reference))
    with torch.inference_mode():
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
    saved = tmp_path / "saved"
    model.save_pretrained(saved, safe_serialization=True)
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = DeepSeekV3QModel.loader.from_pretrained(saved, attn_implementation="eager").eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, expected, rtol=1e-5, atol=1e-6)
    assert not marker.exists()


def test_failed_weight_load_can_retry_without_checkpoint_python(tmp_path, monkeypatch):
    reference, marker = write_checkpoint(tmp_path, tiny_payload("kimi_k2"))
    monkeypatch.setenv(BYPASS_ENV, "1")
    state = checkpoint_state_dict(reference)
    broken = dict(state, **{"model.layers.1.mlp.experts.2.gate_proj.weight": torch.ones(3, 3)})
    save_file(broken, str(tmp_path / "model.safetensors"))
    with pytest.raises((RuntimeError, ValueError), match="automatic conversion of the weights"):
        DeepSeekV3QModel.loader.from_pretrained(tmp_path, trust_remote_code=True)
    assert not marker.exists()
    save_file(state, str(tmp_path / "model.safetensors"))
    model = DeepSeekV3QModel.loader.from_pretrained(tmp_path, trust_remote_code=False, attn_implementation="eager").eval()
    with torch.inference_mode():
        ids = torch.tensor([[1, 7, 8]])
        torch.testing.assert_close(model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_partial_group_kimi_rejected_by_public_config_load_then_same_path_retry(tmp_path, monkeypatch, trust, bypass):
    payload = tiny_payload("kimi_k2", n_group=2, topk_group=1)
    marker = poison_checkpoint(tmp_path, payload)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    with pytest.raises(ValueError, match="Kimi-K2 partial-group routing"):
        load_model_config(tmp_path, trust_remote_code=trust)
    payload["topk_group"] = 2
    poison_checkpoint(tmp_path, payload)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert config.source_model_type == "kimi_k2"
    assert config.model_type == "deepseek_v3"
    assert config.topk_group == config.n_group == 2
    assert not marker.exists()


@pytest.mark.parametrize("via", ["from_config", "from_pretrained", "constructor"])
def test_mutated_kimi_config_is_revalidated_before_model_construction(tmp_path, via):
    write_checkpoint(tmp_path, tiny_payload("kimi_k2"))
    config = load_model_config(tmp_path)
    config.n_group = 2
    before = copy.deepcopy(config.to_dict())
    with pytest.raises(ValueError, match="Kimi-K2 partial-group routing"):
        if via == "from_pretrained":
            DeepSeekV3QModel.loader.from_pretrained(tmp_path, config=config)
        elif via == "from_config":
            DeepSeekV3QModel.loader.from_config(config)
        else:
            from gptqmodel.models.ports.deepseek_v3 import DeepseekV3ForCausalLM as PackagedModel

            PackagedModel(config)
    assert config.to_dict() == before
    config.topk_group = 2
    with torch.device("meta"):
        assert isinstance(DeepSeekV3QModel.loader.from_config(config), DeepseekV3ForCausalLM)


def test_native_deepseek_partial_group_layout_is_not_rejected(tmp_path):
    reference, _ = write_checkpoint(tmp_path, tiny_payload(n_group=2, topk_group=1))
    config = load_model_config(tmp_path)
    assert not hasattr(config, "source_model_type")
    model = DeepSeekV3QModel.loader.from_pretrained(tmp_path, config=config, attn_implementation="eager").eval()
    with torch.inference_mode():
        ids = torch.tensor([[1, 3, 8]])
        torch.testing.assert_close(model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("entry", [GPTQModel.load, GPTQModel.from_pretrained, GPTQModel.from_quantized])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_entrypoints_reject_partial_group_kimi_before_weights_or_python(tmp_path, monkeypatch, entry, trust, bypass):
    marker = poison_checkpoint(tmp_path, tiny_payload("kimi_k2", n_group=2, topk_group=1))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    kwargs = {"quantize_config": None} if entry.__name__ == "from_pretrained" else {}
    with pytest.raises(ValueError, match="Kimi-K2 partial-group routing"):
        entry(str(tmp_path), trust_remote_code=trust, **kwargs)
    assert not marker.exists()


def test_canonical_kimi_provenance_survives_serialization_and_does_not_mutate_input(tmp_path):
    from gptqmodel.models.ports.deepseek_v3 import DeepseekV3Config as PackagedConfig

    payload = tiny_payload("kimi_k2")
    before = copy.deepcopy(payload)
    config = PackagedConfig.from_dict(payload)
    assert payload == before
    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["model_type"] == "deepseek_v3"
    assert saved["source_model_type"] == "kimi_k2"
    saved["n_group"] = 2
    (tmp_path / "config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Kimi-K2 partial-group routing"):
        load_model_config(tmp_path)


@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_kimi_yarn_metadata_and_padded_cache_continuation(tmp_path, q_lora_rank):
    payload = tiny_payload(
        "kimi_k2", q_lora_rank, max_position_embeddings=128,
        rope_scaling={"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4,
                          "beta_fast": 1.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0},
    )
    reference, marker = write_checkpoint(tmp_path, payload)
    config = load_model_config(tmp_path)
    assert config.rope_parameters["factor"] == 32.0
    assert config.rope_parameters["rope_theta"] == 50000.0
    assert config.rope_parameters["beta_fast"] == config.rope_parameters["beta_slow"] == 1.0
    model = DeepSeekV3QModel.loader.from_pretrained(tmp_path, config=config, attn_implementation="eager").eval()
    ids = torch.tensor([[0, 1, 4, 6, 8], [1, 3, 5, 7, 9]])
    mask = (ids != 0).long()
    with torch.inference_mode():
        whole = reference(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(model(ids, attention_mask=mask).logits, whole, rtol=1e-5, atol=1e-6)
        prefix = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        continuation = model(ids[:, 3:], attention_mask=mask, past_key_values=prefix.past_key_values)
        torch.testing.assert_close(continuation.logits, whole[:, 3:], rtol=1e-5, atol=1e-6)
    assert not marker.exists()


@pytest.mark.parametrize("n_group", [1, 2])
def test_kimi_full_group_routing_matches_published_score_rule(tmp_path, n_group):
    poison_checkpoint(tmp_path, tiny_payload("kimi_k2", n_group=n_group, topk_group=n_group))
    config = load_model_config(tmp_path)
    model = DeepSeekV3QModel.loader.from_config(config).eval()
    gate = model.model.layers[1].mlp.gate
    x = torch.linspace(-1.2, 2.4, 3 * config.hidden_size).reshape(1, 3, -1)
    with torch.no_grad():
        gate.weight.copy_(torch.linspace(-2, 1, gate.weight.numel()).reshape_as(gate.weight))
        gate.e_score_correction_bias.copy_(torch.tensor([.25, -.8, .7, -.1]))
        # All groups are selected: Moonshot's masked_fill(0) excludes no scores.
        scores = torch.nn.functional.linear(x.reshape(-1, config.hidden_size), gate.weight).sigmoid()
        indices = (scores + gate.e_score_correction_bias).topk(config.num_experts_per_tok, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * config.routed_scaling_factor
        _, actual_weights, actual_indices = gate(x)
    torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, weights, rtol=0, atol=0)


def test_published_non_decoder_mtp_tensors_survive_safetensors_writer(tmp_path, monkeypatch):
    from safetensors import safe_open

    from tests.test_out_of_model_tensors import (
        _build_writer_with_out_of_model_file,
        _patch_basic_env,
        _patch_streaming,
        _tensor_source,
    )

    source = tmp_path / "source"
    source.mkdir()
    expected = {
        "model.layers.61.enorm.weight": torch.tensor([1.25, -2.5]),
        "model.layers.61.self_attn.q_b_proj.weight": torch.arange(6).reshape(2, 3).float(),
    }
    save_file(expected, str(source / "model.safetensors"))
    writer = _build_writer_with_out_of_model_file(
        str(source), out_of_model_tensor_files=DeepSeekV3QModel.out_of_model_tensors,
    )
    state = {"model.weight": _tensor_source("model.weight", torch.tensor([3.5]))}
    _patch_basic_env(monkeypatch, state)
    _patch_streaming(monkeypatch)
    saved = tmp_path / "saved"
    writer.save_quantized(save_dir=str(saved))
    with safe_open(saved / "model.safetensors", framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {*expected, "model.weight"}
        for name, value in expected.items():
            torch.testing.assert_close(handle.get_tensor(name), value, rtol=0, atol=0)
