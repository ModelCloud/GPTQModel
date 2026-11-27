# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native DeepSeek-V3 and standalone Kimi-K2 checkpoint loading contracts."""

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config as NativeDeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.native.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from gptqmodel.models.ports.native.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config, resolve_trust_remote_code
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
    # Transformers' own class is the numerical oracle the port must reproduce.
    config = NativeDeepseekV3Config.from_dict(dict(payload, model_type="deepseek_v3"))
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = NativeDeepseekV3ForCausalLM(config).eval()
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


@pytest.mark.parametrize("layer_count,mtp_count", [(61, 1), (63, 2), (27, None), (61, 0)])
def test_published_non_decoder_mtp_tensors_survive_safetensors_writer(tmp_path, monkeypatch, layer_count, mtp_count):
    from safetensors import safe_open

    from tests.test_out_of_model_tensors import (
        _build_writer_with_out_of_model_file,
        _patch_basic_env,
        _patch_streaming,
        _tensor_source,
    )

    source = tmp_path / "source"
    source.mkdir()
    expected = {f"model.layers.{index}.enorm.weight": torch.tensor([1.25, -2.5])
                for index in range(layer_count, layer_count + (mtp_count or 0))}
    packed_name = f"model.layers.{layer_count - 1}.self_attn.q_a_proj.qweight"
    save_file({
        **expected,
        "model.layers.61.self_attn.q_a_proj.weight": torch.ones(2),
    }, str(source / "model.safetensors"))
    writer = _build_writer_with_out_of_model_file(str(source))
    writer.model.config.num_hidden_layers = layer_count
    if mtp_count is not None:
        writer.model.config.num_nextn_predict_layers = mtp_count
    type(writer).out_of_model_tensors = DeepSeekV3QModel.out_of_model_tensors
    if layer_count == 61 and mtp_count:
        expected["model.layers.61.self_attn.q_a_proj.weight"] = torch.ones(2)
    state = {packed_name: _tensor_source(packed_name, torch.tensor([3], dtype=torch.int32))}
    _patch_basic_env(monkeypatch, state)
    _patch_streaming(monkeypatch)
    saved = tmp_path / "saved"
    writer.save_quantized(save_dir=str(saved))
    with safe_open(saved / "model.safetensors", framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {*expected, packed_name}
        for name, value in expected.items():
            torch.testing.assert_close(handle.get_tensor(name), value, rtol=0, atol=0)


TOKENIZER_ASSETS = {
    "kimi_k2": (
        "moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50", {
            "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
            "tokenizer_config.json": "5ba77d5822137ac3cb9337d2b84561d10425562cf8b8f2c74c91a832516a778d",
            "chat_template.jinja": "39e8c195b474c1a4148046f41fedd177b2ff8f420fa376e52fcdddb829e8f0c5",
        },
    ),
    "deepseek_v3": (
        "deepseek-ai/DeepSeek-V3", "e815299b0bcbac849fa540c768ef21845365c9eb", {
            "tokenizer.json": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
            "tokenizer_config.json": "637bcd1a08cf7c772ce6a383196b22930921c79c3c73223c5afc0c7f41545546",
        },
    ),
    # Moonlight ships the same byte-identical tiktoken vocabulary and pretokenizer
    # regex as Kimi-K2, but declares its own checkpoint class name.
    "moonlight": (
        "moonshotai/Moonlight-16B-A3B", "476b36a473d4467f94469414bef6cee75c9c8172", {
            "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
            "tokenizer_config.json": "566eff6d3b862587c620edb402dc76417c582605c79e2c9a4bca333a73f44fcb",
        },
    ),
}
TOKENIZER_TEXT = "Hello, world! 你好，世界！"
KIMI_IDS = [19180, 11, 2695, 0, 220, 33845, 378, 2243, 856]
MOONLIGHT_IDS = KIMI_IDS
NATIVE_IDS = [19923, 14, 2058, 3, 223, 30594, 303, 3427, 1175]
KIMI_CHAT = (
    "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>\n"
    "<|im_user|>user<|im_middle|>你好!<|im_end|><|im_assistant|>assistant<|im_middle|>"
)


@pytest.fixture(scope="module")
def real_tokenizer_assets(tmp_path_factory):
    """Only hash-pinned data assets, never checkpoint Python or trained weights."""
    result = {}
    for model_type, (repo_id, revision, files) in TOKENIZER_ASSETS.items():
        path = tmp_path_factory.mktemp(model_type + "_tokenizer")
        for name, digest in files.items():
            source = Path(hf_hub_download(repo_id, name, revision=revision))
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            shutil.copyfile(source, path / name)
        result[model_type] = path
    return result


def tokenizer_checkpoint(path, assets, model_type="kimi_k2", persisted=False):
    path.mkdir(exist_ok=True)
    for source in assets[model_type].iterdir():
        shutil.copyfile(source, path / source.name)
    payload = tiny_payload(
        model_type, vocab_size=163840 if model_type == "kimi_k2" else 129280,
        bos_token_id=163584 if model_type == "kimi_k2" else 0,
        eos_token_id=163585 if model_type == "kimi_k2" else 1, pad_token_id=None,
    )
    if persisted:
        payload.update(model_type="deepseek_v3", source_model_type="kimi_k2")
    marker = poison_checkpoint(path, payload)
    tokenizer_file = path / "tokenizer_config.json"
    config = json.loads(tokenizer_file.read_text())
    if model_type == "kimi_k2":
        config["auto_map"] = {"AutoTokenizer": ["checkpoint_python.TikTokenTokenizer", None]}
    tokenizer_file.write_text(json.dumps(config))
    return marker


def reject_checkpoint_auto(*args, **kwargs):
    raise AssertionError("Tokenizer load must not re-enter checkpoint Auto discovery")


@pytest.mark.parametrize("persisted", [False, True], ids=["raw_alias", "saved_alias"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("path_kind", [str, Path])
def test_real_kimi_tokenizer_default_deny(
    tmp_path, monkeypatch, real_tokenizer_assets, persisted, trust, bypass, path_kind,
):
    from gptqmodel.models.ports.deepseek_v3.tokenization_kimi import TikTokenTokenizer

    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets, persisted=persisted)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_checkpoint_auto)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", reject_checkpoint_auto)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert config.pad_token_id is None
    tokenizer = load_hf_tokenizer(
        path_kind(tmp_path), model_config=config, trust_remote_code=trust,
        code_revision="never-import", config=DeepseekV3Config(), local_files_only=True,
        padding_side="left", model_max_length=137,
    )
    assert type(tokenizer) is TikTokenTokenizer
    assert tokenizer.encode(TOKENIZER_TEXT) == KIMI_IDS
    assert tokenizer.decode(KIMI_IDS) == TOKENIZER_TEXT
    assert tokenizer.pad_token_id == config.pad_token_id == 163839
    assert tokenizer.padding_side == "left" and tokenizer.model_max_length == 137
    assert not marker.exists()


@pytest.mark.parametrize("hostile_tokenizer", [False, True])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_native_deepseek_tokenizer_control(
    tmp_path, monkeypatch, real_tokenizer_assets, trust, bypass, hostile_tokenizer,
):
    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets, "deepseek_v3")
    if hostile_tokenizer:
        metadata_file = tmp_path / "tokenizer_config.json"
        metadata = json.loads(metadata_file.read_text())
        metadata["auto_map"] = {"AutoTokenizer": ["checkpoint_python.TikTokenTokenizer", None]}
        metadata_file.write_text(json.dumps(metadata))
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_checkpoint_auto)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    tokenizer = load_hf_tokenizer(
        tmp_path, model_config=config, trust_remote_code=trust,
        code_revision="never-import", local_files_only=True,
    )
    assert type(tokenizer).__module__.startswith("transformers.")
    assert tokenizer.encode(TOKENIZER_TEXT) == NATIVE_IDS
    assert tokenizer.decode(NATIVE_IDS) == TOKENIZER_TEXT
    assert tokenizer.pad_token_id == config.pad_token_id == 1
    assert not marker.exists()


def test_native_tokenizer_class_name_is_not_kimi_selector(tmp_path, monkeypatch, real_tokenizer_assets):
    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets, "deepseek_v3")
    tokenizer_file = tmp_path / "tokenizer_config.json"
    payload = json.loads(tokenizer_file.read_text())
    payload["tokenizer_class"] = "TikTokenTokenizer"
    tokenizer_file.write_text(json.dumps(payload))
    monkeypatch.setenv(BYPASS_ENV, "1")
    config = load_model_config(tmp_path)
    with pytest.raises(ValueError, match="TikTokenTokenizer|trust_remote_code"):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    payload["tokenizer_class"] = "TokenizersBackend"
    tokenizer_file.write_text(json.dumps(payload))
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    assert tokenizer.encode(TOKENIZER_TEXT) == NATIVE_IDS
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_moonlight_tokenizer_uses_vendored_kimi_family(
    tmp_path, monkeypatch, real_tokenizer_assets, trust, bypass,
):
    """Moonlight declares ``TikTokenTokenizer`` without ``model_type=kimi_k2``.

    Its tiktoken vocabulary and pretokenizer regex are the Kimi-K2 family, so
    the package-owned tokenizer must load it without checkpoint Python.
    """

    from gptqmodel.models.ports.deepseek_v3.tokenization_kimi import TikTokenTokenizer

    for source in real_tokenizer_assets["moonlight"].iterdir():
        shutil.copyfile(source, tmp_path / source.name)
    payload = tiny_payload(
        "deepseek_v3", vocab_size=163840, bos_token_id=163584, eos_token_id=163585, pad_token_id=None,
    )
    marker = poison_checkpoint(tmp_path, payload)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_checkpoint_auto)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", reject_checkpoint_auto)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    tokenizer = load_hf_tokenizer(
        tmp_path, model_config=config, trust_remote_code=trust,
        code_revision="never-import", local_files_only=True,
    )
    assert type(tokenizer) is TikTokenTokenizer
    assert tokenizer.encode(TOKENIZER_TEXT) == MOONLIGHT_IDS
    assert tokenizer.decode(MOONLIGHT_IDS) == TOKENIZER_TEXT
    assert tokenizer.bos_token_id == 163584 and tokenizer.eos_token_id == 163585
    # Moonlight's added_tokens_decoder swaps Kimi-K2's [PAD]/[UNK] ids.
    assert tokenizer.pad_token_id == config.pad_token_id == 163838
    assert tokenizer.unk_token_id == 163839
    assert not marker.exists()


@pytest.fixture
def kimi_tokenizer(tmp_path, real_tokenizer_assets):
    tokenizer_checkpoint(tmp_path, real_tokenizer_assets)
    return load_hf_tokenizer(tmp_path, model_config=load_model_config(tmp_path), local_files_only=True)


@pytest.mark.parametrize(("text", "expected"), [
    (TOKENIZER_TEXT, KIMI_IDS),
    ("\n  résumé café 🧪🚀 1234567", [198, 220, 23606, 7671, 1941, 70591, 17137, 100, 103, 5885, 40101, 220, 6694, 12972, 22]),
    ("[BOS]<|im_user|>hi<|im_end|>[EOS]", [163584, 163587, 9663, 163586, 163585]),
])
def test_kimi_real_tokenizer_exact_ids(kimi_tokenizer, text, expected):
    assert kimi_tokenizer.encode(text) == expected
    assert kimi_tokenizer(text, add_special_tokens=False).input_ids == expected
    assert kimi_tokenizer.decode(expected) == text
    assert kimi_tokenizer.decode(expected, skip_special_tokens=False) == text


def test_kimi_real_tokenizer_batch_specials_and_long_runs(kimi_tokenizer):
    tokenizer = kimi_tokenizer
    assert tokenizer.vocab_size == len(tokenizer) == 163840
    assert len(tokenizer.special_tokens) == 256
    for index in range(163584, 163840):
        token = tokenizer.model.decode_single_token_bytes(index).decode()
        assert tokenizer.encode(token) == [index]
        assert tokenizer.decode([index]) == token
    assert tokenizer.encode("[BOS]", allow_special_tokens=False) != [163584]
    assert tokenizer.decode([163584, 9663, 163585], skip_special_tokens=True) == "hi"
    batch = tokenizer([TOKENIZER_TEXT, "hi"], padding=True, return_tensors="pt", add_special_tokens=False)
    assert batch.input_ids[0].tolist() == KIMI_IDS
    assert batch.input_ids[1].tolist() == [9663] + [163839] * 8
    assert batch.attention_mask.tolist() == [[1] * 9, [1] + [0] * 8]
    assert tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True) == [TOKENIZER_TEXT, "hi"]
    for length in (24_999, 25_000, 25_001):
        text = "x" * length + " " * 25_001 + "漢字"
        assert tokenizer.decode(tokenizer.encode(text)) == text
    text = "a " * 200_001
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_kimi_real_chat_template_and_tools(kimi_tokenizer):
    tokenizer = kimi_tokenizer
    messages = [{"role": "user", "content": "你好!"}]
    assert tokenizer.apply_chat_template(messages, tokenize=False) == KIMI_CHAT
    expected = tokenizer.encode(KIMI_CHAT)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    assert encoded.input_ids.tolist() == [expected]
    assert encoded.attention_mask.tolist() == [[1] * len(expected)]
    assert tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) == KIMI_CHAT.removesuffix(
        "<|im_assistant|>assistant<|im_middle|>",
    )
    tools = [{"type": "function", "function": {
        "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}},
        "name": "f", "description": "D",
    }}]
    original_tools = copy.deepcopy(tools)
    declaration = (
        '[{"function":{"description":"D","name":"f","parameters":{"properties":'
        '{"a":{"type":"integer"},"z":{"type":"string"}},"type":"object"}},"type":"function"}]'
    )
    assert tokenizer.apply_chat_template(messages, tools=tools, tokenize=False) == (
        "<|im_system|>tool_declare<|im_middle|>" + declaration + "<|im_end|>" + KIMI_CHAT
    )
    assert tools == original_tools
    messages += [
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "1"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    assert (
        'ok<|tool_calls_section_begin|><|tool_call_begin|>call_1<|tool_call_argument_begin|>{"x":1}'
        '<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>tool<|im_middle|>'
        '## Return of call_1\n1<|im_end|>'
    ) in rendered


def test_kimi_tokenizer_save_reload_and_failure_retry(tmp_path, monkeypatch, real_tokenizer_assets):
    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets)
    monkeypatch.setenv(BYPASS_ENV, "1")
    config = load_model_config(tmp_path)
    (tmp_path / "tiktoken.model").write_text("not valid base64 BPE\n")
    with pytest.raises(ValueError):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    shutil.copyfile(real_tokenizer_assets["kimi_k2"] / "tiktoken.model", tmp_path / "tiktoken.model")
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    saved = tmp_path / "saved"
    tokenizer.save_pretrained(saved)
    config.save_pretrained(saved)
    assert hashlib.sha256((saved / "tiktoken.model").read_bytes()).hexdigest() == TOKENIZER_ASSETS["kimi_k2"][2]["tiktoken.model"]
    assert load_model_config(saved).source_model_type == "kimi_k2"
    reloaded = load_hf_tokenizer(saved, model_config=load_model_config(saved), local_files_only=True)
    assert reloaded.encode(TOKENIZER_TEXT) == KIMI_IDS
    assert reloaded.apply_chat_template([{"role": "user", "content": "你好!"}], tokenize=False) == KIMI_CHAT
    assert not marker.exists()


def test_public_model_load_with_actual_kimi_tokenizer(tmp_path, monkeypatch, real_tokenizer_assets):
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    payload = tiny_payload("kimi_k2", vocab_size=163840, bos_token_id=163584, eos_token_id=163585, pad_token_id=None)
    reference, marker = write_checkpoint(tmp_path, payload)
    tokenizer_checkpoint(tmp_path, real_tokenizer_assets)
    model = GPTQModel.load(
        str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=False,
        attn_implementation="eager", local_files_only=True,
    )
    tokenizer = model.tokenizer
    assert tokenizer.encode(TOKENIZER_TEXT) == KIMI_IDS
    batch = tokenizer(TOKENIZER_TEXT, return_tensors="pt")
    with torch.inference_mode():
        torch.testing.assert_close(
            model.model(**batch).logits, reference(**batch).logits, rtol=1e-5, atol=1e-6,
        )
    assert not marker.exists()


def test_kimi_tokenizer_regression_without_owned_factory(tmp_path, monkeypatch, real_tokenizer_assets):
    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    monkeypatch.setattr(DeepSeekV3QModel, "loader", replace(DeepSeekV3QModel.loader, tokenizer_class=None))
    with pytest.raises(ValueError, match="TikTokenTokenizer"):
        load_hf_tokenizer(tmp_path, model_config=load_model_config(tmp_path), local_files_only=True)
    assert not marker.exists()


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
def test_tokenizer_factory_preserves_subfolder_and_source(tmp_path, real_tokenizer_assets, model_type):
    marker = tokenizer_checkpoint(tmp_path / "nested", real_tokenizer_assets, model_type)
    config = load_model_config(str(tmp_path), subfolder="nested")
    tokenizer = load_hf_tokenizer(
        str(tmp_path), model_config=config, subfolder="nested", local_files_only=True,
    )
    assert tokenizer.encode(TOKENIZER_TEXT) == (KIMI_IDS if model_type == "kimi_k2" else NATIVE_IDS)
    assert not marker.exists()


def test_native_non_bytelevel_fallback_rejects_remote_ambiguity_and_retries(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"<unk>": 0, "hello": 1, "<pad>": 2}, unk_token="<unk>"))
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>").save_pretrained(tmp_path)
    marker = poison_checkpoint(tmp_path, tiny_payload())
    metadata_file = tmp_path / "tokenizer_config.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["auto_map"] = {"AutoTokenizer": ["checkpoint_python.RemoteTokenizer", None]}
    metadata_file.write_text(json.dumps(metadata))
    monkeypatch.setenv(BYPASS_ENV, "1")
    monkeypatch.setattr(AutoConfig, "from_pretrained", reject_checkpoint_auto)
    config = load_model_config(tmp_path)
    with pytest.raises(ValueError, match="Unqualified native DeepSeek tokenizer metadata"):
        load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    metadata.pop("auto_map")
    metadata_file.write_text(json.dumps(metadata))
    tokenizer = load_hf_tokenizer(tmp_path, model_config=config, trust_remote_code=True, local_files_only=True)
    assert tokenizer.encode("hello") == [1]
    assert not marker.exists()


def test_native_tokenizer_rejects_foreign_global_registration(tmp_path, monkeypatch, real_tokenizer_assets):
    from transformers import PreTrainedConfig, PreTrainedTokenizerFast
    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

    class ForeignConfig(PreTrainedConfig):
        model_type = "foreign_tokenizer_fixture"

    class ForeignTokenizer(PreTrainedTokenizerFast):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("A registered third-party tokenizer must not replace the native implementation")

    marker = tokenizer_checkpoint(tmp_path, real_tokenizer_assets, "deepseek_v3")
    metadata_file = tmp_path / "tokenizer_config.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["tokenizer_class"] = "ForeignTokenizer"
    metadata_file.write_text(json.dumps(metadata))
    monkeypatch.setitem(TOKENIZER_MAPPING._extra_content, ForeignConfig, (ForeignTokenizer, None))
    monkeypatch.setenv(BYPASS_ENV, "1")
    with pytest.raises(ValueError, match="ForeignTokenizer.*not available natively"):
        load_hf_tokenizer(tmp_path, model_config=load_model_config(tmp_path), trust_remote_code=True, local_files_only=True)
    assert not marker.exists()
