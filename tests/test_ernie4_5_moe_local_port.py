# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Transformers 5 ERNIE MoE qualification; no checkpoint Python."""

import copy
import json

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ernie4_5_moe import Ernie4_5_MoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(**overrides):
    kwargs = {
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 3,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
        "moe_intermediate_size": 12, "moe_k": 2, "moe_num_experts": 3,
        "moe_num_shared_experts": 1, "moe_layer_start_index": 1, "tie_word_embeddings": False,
    }
    kwargs.update(overrides)
    config = Ernie4_5_MoeConfig(**kwargs)
    config._attn_implementation = "eager"
    return config


def tiny_model(**overrides):
    torch.manual_seed(42)
    return Ernie4_5_MoeForCausalLM(tiny_config(**overrides)).eval()


def checkpoint(path, **overrides):
    model = tiny_model(**overrides)
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}, unk_token="<unk>")),
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "untrusted.RemoteConfig",
        "AutoModelForCausalLM": "untrusted.RemoteModel",
    }
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "untrusted.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise AssertionError('checkpoint Python must never execute')\n"
    )
    return model, marker


def test_exact_native_registration_needs_no_legacy_forward_or_remote_permission():
    assert MODEL_MAP["ernie4_5_moe"] is Ernie4_5_MoeQModel
    assert AutoModelForCausalLM._model_mapping[Ernie4_5_MoeConfig] is Ernie4_5_MoeForCausalLM
    assert not Ernie4_5_MoeQModel.require_trust_remote_code
    assert not Ernie4_5_MoeQModel.require_monkeypatch


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_ignores_malicious_auto_map_and_preserves_logits(tmp_path, monkeypatch, trust, bypass):
    reference, marker = checkpoint(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security control under test.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is Ernie4_5_MoeConfig
    loaded = GPTQModel.load(
        str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32,
        attn_implementation="eager",
    )
    assert isinstance(loaded, Ernie4_5_MoeQModel)
    assert type(loaded.model) is Ernie4_5_MoeForCausalLM
    assert loaded.trust_remote_code is False
    ids = torch.tensor([[1, 7, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-6, atol=1e-6)
    assert not marker.exists()


@pytest.mark.parametrize("shared,bias,interval,start", [(0, False, 1, 1), (1, True, 1, 0), (2, False, 2, 1)])
def test_defusion_preserves_exact_expert_tensors_and_native_outputs(shared, bias, interval, start):
    model = tiny_model(moe_num_shared_experts=shared, use_bias=bias, moe_layer_interval=interval,
                       moe_layer_start_index=start)
    source = copy.deepcopy(model.state_dict())
    ids = torch.tensor([[1, 7, 8, 9], [1, 4, 5, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with torch.inference_mode():
        baseline = model(ids, attention_mask=mask, output_router_logits=True)
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}.mlp.experts."
        if prefix + "gate_up_proj" not in source:
            continue
        for expert in range(model.config.moe_num_experts):
            gate, up = source[prefix + "gate_up_proj"][expert].chunk(2, dim=0)
            for name, expected in (("gate_proj", gate), ("up_proj", up),
                                   ("down_proj", source[prefix + "down_proj"][expert])):
                actual = layer.get_submodule(f"mlp.experts.{expert}.{name}")
                torch.testing.assert_close(actual.weight, expected, rtol=0, atol=0)
        assert not hasattr(layer.mlp.experts, "gate_up_proj")
        assert not hasattr(layer.mlp.experts, "down_proj")
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, output_router_logits=True)
    torch.testing.assert_close(actual.logits, baseline.logits, rtol=1e-6, atol=1e-6)
    for expected, result in zip(baseline.router_logits, actual.router_logits, strict=True):
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
        assert torch.equal(result.topk(model.config.moe_k).indices, expected.topk(model.config.moe_k).indices)


@pytest.mark.parametrize("shared", [0, 2])
def test_native_cache_continuation_generation_and_safetensors_reload(tmp_path, shared):
    model = tiny_model(moe_num_shared_experts=shared)
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, model, cleanup_original=True)
    ids = torch.tensor([[1, 7, 8, 9], [1, 4, 5, 6]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :3], use_cache=True)
        assert prefix.past_key_values.get_seq_length() == 3
        suffix = model(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
        assert suffix.past_key_values.get_seq_length() == 4
        torch.testing.assert_close(suffix.logits, full[:, 3:], rtol=1e-5, atol=1e-6)
        cached = model.generate(ids, min_new_tokens=3, max_new_tokens=3, do_sample=False, use_cache=True)
        uncached = model.generate(ids, min_new_tokens=3, max_new_tokens=3, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))
    restored = Ernie4_5_MoeForCausalLM.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert _convert_model_with_defuser(Ernie4_5_MoeQModel, restored, cleanup_original=True)
    assert model.state_dict().keys() == restored.state_dict().keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids).logits, full, rtol=1e-6, atol=1e-6)
        assert torch.equal(restored.generate(ids, min_new_tokens=3, max_new_tokens=3, do_sample=False), cached)


def test_failed_checkpoint_read_can_retry_without_executing_remote_code(tmp_path, monkeypatch):
    reference, marker = checkpoint(tmp_path)
    monkeypatch.delenv(BYPASS_ENV, raising=False)
    weights = {name: value.clone() for name, value in load_file(tmp_path / "model.safetensors").items()}
    (tmp_path / "model.safetensors").write_bytes(b"invalid header")
    with pytest.raises(SafetensorError, match="(?i)(header|safetensor)"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    save_file(weights, tmp_path / "model.safetensors", metadata={"format": "pt"})
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    ids = torch.tensor([[1, 4, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-6, atol=1e-6)
    assert not marker.exists()


def test_original_numbered_experts_and_legacy_statics_checkpoint_load_exactly(tmp_path):
    reference, marker = checkpoint(tmp_path)
    with torch.no_grad():
        for layer in reference.model.layers[1:]:
            layer.mlp.gate.moe_statics.e_score_correction_bias.copy_(torch.tensor([[0.2, -0.15, 0.08]]))
    original = {}
    for name, tensor in reference.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for index in range(reference.config.moe_num_experts):
                gate, up = tensor[index].chunk(2, dim=0)
                original[f"{prefix}{index}.gate_proj.weight"] = gate.clone()
                original[f"{prefix}{index}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for index in range(reference.config.moe_num_experts):
                original[f"{prefix}{index}.down_proj.weight"] = tensor[index].clone()
        else:
            original[name.replace(".mlp.gate.moe_statics.", ".mlp.moe_statics.")] = tensor.clone()
    save_file(original, tmp_path / "model.safetensors", metadata={"format": "pt"})
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, attn_implementation="eager")
    assert loaded.model.state_dict().keys() == reference.state_dict().keys()
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected, rtol=0, atol=0)
    ids = torch.tensor([[1, 5, 8, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_awq_scales_and_quantizes_every_native_projection_and_preserves_routers(tmp_path, caplog, dtype):
    source = tmp_path / "source"
    reference, marker = checkpoint(source, num_hidden_layers=2, intermediate_size=32, moe_intermediate_size=16)
    router = {
        name: tensor.clone() for name, tensor in reference.state_dict().items()
        if ".mlp.gate." in name
    }
    qcfg = AWQConfig(bits=4, group_size=16, device="cpu", offload_to_disk=False)
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH,
                          dtype=dtype, attn_implementation="eager")
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]])
    model.quantize(
        CalibrationConfig(dataset=[{"input_ids": ids, "attention_mask": torch.ones_like(ids)}], batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name: module for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected_names = {
        f"model.layers.{layer}.self_attn.{projection}_proj" for layer in range(2) for projection in "qkvo"
    }
    expected_names.update(f"model.layers.0.mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj"))
    expected_names.update(f"model.layers.1.mlp.shared_experts.{projection}"
                          for projection in ("gate_proj", "up_proj", "down_proj"))
    expected_names.update(f"model.layers.1.mlp.experts.{index}.{projection}"
                          for index in range(3) for projection in ("gate_proj", "up_proj", "down_proj"))
    assert len(packed) == 23 and set(packed) == expected_names
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    assert "no AWQ scaling groups were generated" not in caplog.text
    searched = {(int(row["layer"]), str(row["module"])) for row in model.quant_log if row["loss"] != "not applicable"}
    # Scale search ran on both the dense prefix and the routed/shared MoE layer.
    assert {(0, "self_attn.q_proj"), (0, "mlp.gate_proj"), (1, "self_attn.q_proj"),
            (1, "mlp.shared_experts.gate_proj")} <= searched
    assert not marker.exists()
    for name, expected in router.items():
        actual = model.model.state_dict()[name]
        assert actual.shape == expected.shape and actual.dtype == torch.float32, name
        if name.endswith("e_score_correction_bias"):
            # The routing bias is added after softmax and never sees the scale.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            # The fp32 router stays dense; it only absorbs the folded input scale.
            assert torch.isfinite(actual).all() and not torch.equal(actual, expected), name
    with torch.inference_mode():
        logits = model.model(ids).logits
    assert torch.isfinite(logits).all()
    destination = tmp_path / "quantized"
    model.save(str(destination))
    assert list(destination.glob("*.safetensors"))
    assert not list(destination.glob("*.bin"))
    restored = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.AWQ_TORCH,
                             dtype=dtype, attn_implementation="eager")
    assert restored.model.state_dict().keys() == model.model.state_dict().keys()
    for name, value in model.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0, msg=name)
    with torch.inference_mode():
        torch.testing.assert_close(restored.model(ids).logits, logits, rtol=1e-3, atol=1e-3)
