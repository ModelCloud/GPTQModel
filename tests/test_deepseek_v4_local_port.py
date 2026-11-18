# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native DeepSeek V4 qualification without checkpoint Python or trained weights."""

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.native.auto import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
from gptqmodel.models.ports.native.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from gptqmodel.models.ports.native.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


ATTENTION_TYPES = ("sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention")
MLP_TYPES = ("hash_moe", "moe")


def tiny_config(attention="sliding_attention", mlp="moe", **overrides):
    values = {
        "vocab_size": 32, "hidden_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "q_lora_rank": 8, "n_routed_experts": 3, "num_experts_per_tok": 2,
        "hc_mult": 2, "hc_sinkhorn_iters": 4, "o_groups": 2, "o_lora_rank": 8,
        "sliding_window": 4, "layer_types": [attention], "mlp_layer_types": [mlp],
        "compress_rates": {"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        "index_n_heads": 2, "index_head_dim": 8, "index_topk": 2,
        "partial_rotary_factor": 0.5, "max_position_embeddings": 64,
        "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
    }
    values.update(overrides)
    config = DeepseekV4Config(**values)
    config._attn_implementation = "eager"
    return config


def native_model(attention="sliding_attention", mlp="moe", **overrides):
    torch.manual_seed(317)
    model = DeepseekV4ForCausalLM(tiny_config(attention, mlp, **overrides)).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                gate.tid2eid.copy_(torch.arange(64).reshape(32, 2) % 3)
            else:
                gate.e_score_correction_bias.copy_(torch.tensor([0.03, -0.01, 0.02]))
    return model


def save_tokenizer(path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3}, unk_token="[UNK]",
        )),
        pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)


@pytest.mark.parametrize("offload", [False, True])
def test_public_fp16_awq_with_fp32_compressor(tmp_path, offload):
    model_dir = tmp_path / "source"
    model = native_model("compressed_sparse_attention", "hash_moe")
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.75, 1.25, model.config.hidden_size))
    model.save_pretrained(model_dir)
    save_tokenizer(model_dir)
    wrapper = GPTQModel.load(
        str(model_dir),
        dtype=torch.float16,
        quantize_config=AWQConfig(
            device="cpu", format=FORMAT.GEMM, group_size=-1, fallback=None,
            offload_to_disk=offload, offload_to_disk_path=str(tmp_path / "offload"),
        ),
    )
    ids = [2, 3, 4, 5, 6, 7, 8, 9]
    wrapper.quantize(CalibrationConfig(dataset=[{"input_ids": ids}], batch_size=1, min_length=1))
    layer = wrapper.model.model.layers[0]
    assert layer.self_attn.compressor.kv_proj.weight.dtype == torch.float32
    assert layer.self_attn.compressor.indexer.scorer.weights_proj.weight.dtype == torch.float32
    assert layer.mlp.experts[0].gate_proj.qweight.numel() > 0

    input_ids = torch.tensor([ids])
    with torch.inference_mode():
        expected = wrapper.model(input_ids, use_cache=False).logits
    assert torch.isfinite(expected).all()

    packed_dir = tmp_path / "packed"
    wrapper.save(str(packed_dir))
    reloaded = GPTQModel.load(str(packed_dir), dtype=torch.float16, device="cpu")
    reloaded_layer = reloaded.model.model.layers[0]
    torch.testing.assert_close(reloaded.model.model.norm.weight, wrapper.model.model.norm.weight, rtol=0, atol=0)
    for name in ("self_attn.compressor.kv_proj", "self_attn.compressor.indexer.scorer.weights_proj"):
        torch.testing.assert_close(
            reloaded_layer.get_submodule(name).weight, layer.get_submodule(name).weight, rtol=0, atol=0,
        )
    torch.testing.assert_close(
        reloaded_layer.mlp.experts[0].gate_proj.qweight, layer.mlp.experts[0].gate_proj.qweight, rtol=0, atol=0,
    )
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(input_ids, use_cache=False).logits, expected, rtol=0, atol=0)


def source_checkpoint(model):
    """Spell independent tensors in the released DeepSeek inference namespace."""
    result = {}
    for key, value in model.state_dict().items():
        key = key.removeprefix("model.")
        if key.endswith(".experts.gate_up_proj"):
            prefix = key.removesuffix("gate_up_proj").replace(".mlp.", ".ffn.")
            for index, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                result[f"{prefix}{index}.w1.weight"] = gate.clone()
                result[f"{prefix}{index}.w3.weight"] = up.clone()
            continue
        if key.endswith(".experts.down_proj"):
            prefix = key.removesuffix("down_proj").replace(".mlp.", ".ffn.")
            for index, expert in enumerate(value):
                result[f"{prefix}{index}.w2.weight"] = expert.clone()
            continue
        key = {
            "embed_tokens.weight": "embed.weight", "lm_head.weight": "head.weight",
            "hc_head.hc_fn": "hc_head_fn", "hc_head.hc_base": "hc_head_base", "hc_head.hc_scale": "hc_head_scale",
        }.get(key, key)
        for old, new in (
            (".attn_hc.fn", ".hc_attn_fn"), (".attn_hc.base", ".hc_attn_base"), (".attn_hc.scale", ".hc_attn_scale"),
            (".ffn_hc.fn", ".hc_ffn_fn"), (".ffn_hc.base", ".hc_ffn_base"), (".ffn_hc.scale", ".hc_ffn_scale"),
            (".input_layernorm.", ".attn_norm."), (".post_attention_layernorm.", ".ffn_norm."),
            (".self_attn.compressor.indexer.scorer.weights_proj.", ".attn.indexer.weights_proj."),
            (".self_attn.compressor.indexer.q_b_proj.", ".attn.indexer.wq_b."),
            (".self_attn.compressor.indexer.", ".attn.indexer.compressor."),
            (".self_attn.", ".attn."), (".mlp.", ".ffn."),
            (".sinks", ".attn_sink"), (".q_a_proj.", ".wq_a."), (".q_b_proj.", ".wq_b."),
            (".q_a_norm.", ".q_norm."), (".kv_proj.", ".wkv."),
            (".o_a_proj.", ".wo_a."), (".o_b_proj.", ".wo_b."),
            (".compressor.kv_norm.", ".compressor.norm."), (".position_bias", ".ape"),
            (".compressor.gate_proj.", ".compressor.wgate."),
            (".gate.e_score_correction_bias", ".gate.bias"),
            (".shared_experts.gate_proj.", ".shared_experts.w1."),
            (".shared_experts.up_proj.", ".shared_experts.w3."),
            (".shared_experts.down_proj.", ".shared_experts.w2."),
        ):
            key = key.replace(old, new)
        assert key not in result
        result[key] = value.clone()
    return result


@pytest.fixture
def checkpoint(tmp_path):
    model = native_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    save_tokenizer(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "malicious_config.EvilConfig",
        "AutoModelForCausalLM": "malicious_model.EvilModel",
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    for name in ("malicious_config", "malicious_model"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    return tmp_path, model


def test_v4_uses_exact_native_mapping_without_inherited_v3_trust():
    assert {key for key, value in MODEL_MAP.items() if value is DeepSeekV4QModel} == {"deepseek_v4"}
    assert not DeepSeekV4QModel.require_trust_remote_code
    assert isinstance(DeepSeekV4QModel.loader, LocalModelLoader)
    assert CONFIG_MAPPING["deepseek_v4"] is DeepseekV4Config
    assert MODEL_FOR_CAUSAL_LM_MAPPING["deepseek_v4"] is DeepseekV4ForCausalLM
    assert isinstance(DeepSeekV4QModel.loader.from_config(tiny_config()), DeepseekV4ForCausalLM)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_ignores_checkpoint_python(checkpoint, monkeypatch, trust, bypass):
    path, reference = checkpoint
    if bypass:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    else:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    assert resolve_trust_remote_code(str(path), trust_remote_code=trust) is False
    assert type(load_model_config(str(path), trust_remote_code=trust)) is DeepseekV4Config
    loaded = GPTQModel.load(str(path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is DeepSeekV4QModel
    assert type(loaded.model) is DeepseekV4ForCausalLM
    assert not loaded.trust_remote_code
    ids = torch.tensor([[2, 4, 5, 7, 8, 6]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("attention", ATTENTION_TYPES)
@pytest.mark.parametrize("mlp", MLP_TYPES)
def test_native_cache_generation_defusion_and_safetensors(tmp_path, attention, mlp):
    reference = native_model(attention, mlp)
    source = tmp_path / "source"
    source.mkdir()
    reference.config.save_pretrained(source)
    save_file(source_checkpoint(reference), source / "model.safetensors", metadata={"format": "pt"})
    converted, info = DeepSeekV4QModel.loader.from_pretrained(source, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert converted.state_dict().keys() == reference.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(converted.state_dict()[key], value, rtol=0, atol=0)
    ids = torch.tensor([[2, 4, 6, 9, 5, 8, 7]])
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
        torch.testing.assert_close(converted(ids, use_cache=False).logits, expected, rtol=0, atol=0)
        assert expected.std() > 0.01
        prefill = reference(ids[:, :3], use_cache=True)
        outputs = []
        cache = prefill.past_key_values
        for index in range(3, ids.shape[1]):
            result = reference(ids[:, index:index + 1], past_key_values=cache, use_cache=True)
            outputs.append(result.logits)
            cache = result.past_key_values
        assert cache.get_seq_length() == ids.shape[1]
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected[:, 3:], rtol=1e-5, atol=1e-6)
        cached = reference.generate(ids[:, :3], max_new_tokens=3, do_sample=False, use_cache=True, eos_token_id=None)
        uncached = reference.generate(ids[:, :3], max_new_tokens=3, do_sample=False, use_cache=False, eos_token_id=None)
        assert cached.shape == (1, 6)
        assert torch.equal(cached, uncached)
        perturbed = ids.clone()
        perturbed[:, -2:] = torch.tensor([[11, 13]])
        torch.testing.assert_close(reference(perturbed, use_cache=False).logits[:, :-2], expected[:, :-2], rtol=0, atol=0)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("*.bin")) and not list(tmp_path.glob("*.pt"))
    reloaded, info = DeepSeekV4QModel.loader.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    reloaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
    second = tmp_path / "again"
    reloaded.save_pretrained(second, safe_serialization=True)
    final = DeepSeekV4QModel.loader.from_pretrained(second).eval()
    for key, value in reloaded.state_dict().items():
        torch.testing.assert_close(final.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(final(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
        assert _convert_model_with_defuser(DeepSeekV4QModel, final, cleanup_original=True)
        torch.testing.assert_close(final(ids, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)


def test_failed_native_load_retries_without_remote_dispatch(checkpoint):
    path, reference = checkpoint
    tensors = {key: value.clone() for key, value in reference.state_dict().items()}
    key = "model.layers.0.self_attn.q_a_proj.weight"
    bad = dict(tensors)
    bad[key] = bad[key][:1]
    save_file(bad, path / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises((RuntimeError, ValueError)):
        GPTQModel.load(str(path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    loaded = GPTQModel.load(str(path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    ids = torch.tensor([[2, 5, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, reference(ids).logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mlp", MLP_TYPES)
def test_native_and_defused_moe_match_source_routing_and_clamping(mlp):
    model = native_model(mlp=mlp)
    block = model.model.layers[0].mlp
    hidden = torch.randn(2, 5, 16) * 2
    ids = torch.tensor([[2, 4, 5, 7, 8], [6, 9, 10, 11, 12]])
    with torch.no_grad():
        block.gate.weight.normal_(std=0.3)
        block.experts.gate_up_proj.normal_(std=3)
        block.shared_experts.gate_proj.weight.normal_(std=3)
        block.shared_experts.up_proj.weight.normal_(std=3)
    source = source_checkpoint(model)
    flat = hidden.flatten(0, 1)
    scores = F.softplus(F.linear(flat, source["layers.0.ffn.gate.weight"])).sqrt()
    if mlp == "hash_moe":
        selected = source["layers.0.ffn.gate.tid2eid"][ids.flatten()]
    else:
        selected = (scores + source["layers.0.ffn.gate.bias"]).topk(2, dim=-1, sorted=False).indices
    assert selected.unique().numel() == 3
    chosen = scores.gather(-1, selected)
    chosen = chosen / chosen.sum(-1, keepdim=True) * model.config.routed_scaling_factor
    clamped_count = 0

    def expert(value, name):
        nonlocal clamped_count
        prefix = "layers.0.ffn." + name
        gate = F.linear(value, source[prefix + ".w1.weight"])
        up = F.linear(value, source[prefix + ".w3.weight"])
        limit = model.config.swiglu_limit
        clamped_count += int((gate > limit).sum() + (up.abs() > limit).sum())
        return F.linear(F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit), source[prefix + ".w2.weight"])

    expected = expert(flat, "shared_experts")
    for row in range(flat.shape[0]):
        for slot in range(2):
            expected[row] += chosen[row, slot] * expert(flat[row], f"experts.{selected[row, slot].item()}")
    expected = expected.view_as(hidden)
    assert clamped_count > 0
    with torch.inference_mode():
        torch.testing.assert_close(block(hidden, input_ids=ids), expected, rtol=1e-5, atol=1e-5)
        assert _convert_model_with_defuser(DeepSeekV4QModel, model, cleanup_original=True)
        torch.testing.assert_close(block(hidden, input_ids=ids), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_native_layers_and_grouped_head_variant(tmp_path, dtype):
    reference = native_model(
        num_hidden_layers=3, layer_types=list(ATTENTION_TYPES),
        mlp_layer_types=["hash_moe", "moe", "moe"], hc_mult=4,
        num_attention_heads=4, o_groups=1, q_lora_rank=12,
    )
    reference.save_pretrained(tmp_path, safe_serialization=True)
    model = DeepSeekV4QModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
    owner = object.__new__(DeepSeekV4QModel)
    owner.after_model_load(model)
    ids = torch.tensor([[2, 4, 6, 9, 5, 8, 7, 12]])
    tolerance = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 2e-3, "atol": 2e-3}
    with torch.inference_mode():
        expected = model(ids, use_cache=False).logits
        assert torch.isfinite(expected).all() and expected.std() > 0.01
        prefill = model(ids[:, :3], use_cache=True)
        logits = []
        cache = prefill.past_key_values
        for index in range(3, ids.shape[1]):
            step = model(ids[:, index:index + 1], use_cache=True, past_key_values=cache)
            cache = step.past_key_values
            logits.append(step.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), expected[:, 3:], **tolerance)
        _convert_model_with_defuser(DeepSeekV4QModel, model, cleanup_original=True)
        torch.testing.assert_close(model(ids, use_cache=False).logits, expected, **tolerance)
