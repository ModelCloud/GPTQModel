# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Afmoe qualification; no checkpoint Python or trained weights required."""

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.nn import functional as F
from transformers import AfmoeConfig, AfmoeForCausalLM, AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.afmoe import AfMoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.native.afmoe.configuration_afmoe import AfmoeConfig as VendoredAfmoeConfig
from gptqmodel.models.ports.native.afmoe.modeling_afmoe import AfmoeForCausalLM as VendoredAfmoeForCausalLM
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_dense_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_experts": 3,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "global_attn_every_n_layers": 2,
        "sliding_window": 4,
        "max_position_embeddings": 32,
        "mup_enabled": True,
        "route_scale": 2.826,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "score_func": "sigmoid",
        "route_norm": True,
    }
    values.update(overrides)
    config = AfmoeConfig(**values)
    config._attn_implementation = "eager"
    return config


def split_expert_checkpoint(model):
    """Independently express native packed tensors in released Trinity layout."""
    result = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for index, expert in enumerate(tensor):
                gate, up = expert.chunk(2, dim=0)
                result[f"{prefix}.{index}.gate_proj.weight"] = gate.clone()
                result[f"{prefix}.{index}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for index, expert in enumerate(tensor):
                result[f"{prefix}.{index}.down_proj.weight"] = expert.clone()
        else:
            result[name] = tensor.clone()
    return result


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(842)
    model = AfmoeForCausalLM(tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "malicious_config.EvilConfig",
        "AutoModelForCausalLM": "malicious_model.EvilModel",
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    for name in ("malicious_config", "malicious_model"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    return tmp_path, model


def test_afmoe_is_native_without_a_remote_code_requirement():
    assert {key for key, value in MODEL_MAP.items() if value is AfMoeQModel} == {"afmoe"}
    assert not AfMoeQModel.require_trust_remote_code
    assert AutoConfig.for_model("afmoe").__class__ is AfmoeConfig
    assert AutoModelForCausalLM._model_mapping[AfmoeConfig] is AfmoeForCausalLM


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_ignores_malicious_auto_map(checkpoint, monkeypatch, trust, bypass):
    path, reference = checkpoint
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    assert resolve_trust_remote_code(str(path), trust_remote_code=trust) is False
    assert type(load_model_config(str(path), trust_remote_code=trust)) is VendoredAfmoeConfig
    loaded = GPTQModel.load(str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is AfMoeQModel
    assert type(loaded.model) is VendoredAfmoeForCausalLM
    assert not loaded.trust_remote_code
    ids = torch.tensor([[2, 4, 5, 7, 8, 6]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_attention_heads": 4, "num_key_value_heads": 2, "route_scale": 2.448},
        {"num_dense_layers": 0, "num_shared_experts": 2, "mup_enabled": False},
        {"num_dense_layers": 2, "mup_enabled": False},
    ],
)
def test_native_conversion_cache_generation_and_safetensors(tmp_path, overrides):
    torch.manual_seed(73)
    reference = AfmoeForCausalLM(tiny_config(**overrides)).eval()
    # Make routing bias nonzero so conversion cannot silently discard it.
    with torch.no_grad():
        for layer in reference.model.layers:
            if layer.moe_enabled:
                layer.mlp.expert_bias.copy_(torch.tensor([0.03, -0.01, 0.02]))
    source = split_expert_checkpoint(reference)
    reference.config.save_pretrained(tmp_path)
    save_file(source, tmp_path / "model.safetensors", metadata={"format": "pt"})
    loaded, info = AfMoeQModel.loader.from_pretrained(tmp_path, output_loading_info=True)
    loaded.eval()
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert loaded.state_dict().keys() == reference.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    ids = torch.tensor([[2, 3, 9, 7, 5, 4, 6]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        assert expected.std() > 0.01
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
        prefill = loaded(ids[:, :3], use_cache=True)
        continued = loaded(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)
        torch.testing.assert_close(continued.logits, expected[:, 3:], rtol=1e-5, atol=1e-6)
        cached = loaded.generate(ids[:, :3], max_new_tokens=3, do_sample=False, use_cache=True, eos_token_id=None)
        uncached = loaded.generate(ids[:, :3], max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
        assert cached.shape == (1, 6)
        assert torch.equal(cached, uncached)
        before = loaded(ids, use_cache=False).logits
        _convert_model_with_defuser(AfMoeQModel, loaded, cleanup_original=True)
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, before, rtol=1e-5, atol=1e-6)
    assert loaded.state_dict().keys() == source.keys()
    for key, value in source.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    output = tmp_path / "resaved"
    loaded.save_pretrained(output, safe_serialization=True)
    assert (output / "model.safetensors").is_file()
    assert not list(output.glob("*.bin")) and not list(output.glob("*.pt"))
    reloaded = AfMoeQModel.loader.from_pretrained(output).eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids).logits, expected, rtol=1e-5, atol=1e-6)
    second_output = tmp_path / "resaved_again"
    reloaded.save_pretrained(second_output, safe_serialization=True)
    final = AfMoeQModel.loader.from_pretrained(second_output).eval()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(final.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(final(ids).logits, expected, rtol=1e-5, atol=1e-6)


def test_native_and_defused_moe_match_released_routing_equations():
    torch.manual_seed(73)
    model = AfmoeForCausalLM(tiny_config()).eval()
    block = model.model.layers[1].mlp
    hidden = torch.randn(2, 5, 16)
    with torch.no_grad():
        block.expert_bias.copy_(torch.tensor([0.02, -0.01, 0.01]))
        block.router.gate.weight.normal_(std=0.2)
    weights = split_expert_checkpoint(model)
    prefix = "model.layers.1.mlp."

    def mlp(value, name):
        gate = F.linear(value, weights[prefix + name + ".gate_proj.weight"])
        up = F.linear(value, weights[prefix + name + ".up_proj.weight"])
        return F.linear(F.silu(gate) * up, weights[prefix + name + ".down_proj.weight"])

    flat = hidden.flatten(0, 1)
    scores = torch.sigmoid(F.linear(flat, weights[prefix + "router.gate.weight"]).float())
    selected = (scores + weights[prefix + "expert_bias"]).topk(2, dim=-1).indices
    assert selected.unique().numel() == 3
    chosen = scores.gather(-1, selected)
    chosen = chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * model.config.route_scale
    expected = mlp(flat, "shared_experts")
    for row in range(flat.shape[0]):
        for slot in range(2):
            expert = selected[row, slot].item()
            expected[row] += chosen[row, slot] * mlp(flat[row], f"experts.{expert}")
    expected = expected.view_as(hidden)
    with torch.inference_mode():
        torch.testing.assert_close(block(hidden), expected, rtol=1e-5, atol=1e-7)
        _convert_model_with_defuser(AfMoeQModel, model, cleanup_original=True)
        torch.testing.assert_close(block(hidden), expected, rtol=1e-5, atol=1e-7)


def test_failed_native_load_can_retry_without_remote_dispatch(checkpoint):
    path, reference = checkpoint
    tensors = split_expert_checkpoint(reference)
    key = "model.layers.1.mlp.experts.2.down_proj.weight"
    bad = dict(tensors)
    bad[key] = bad[key][:1]
    save_file(bad, path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises((RuntimeError, ValueError)):
        AfMoeQModel.loader.from_pretrained(path, trust_remote_code=False)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    loaded = AfMoeQModel.loader.from_pretrained(path, trust_remote_code=False).eval()
    ids = torch.tensor([[2, 5, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)
