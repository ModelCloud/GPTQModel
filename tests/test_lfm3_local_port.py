# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native LFM3 qualification; no checkpoint Python or trained weights needed."""

import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import AutoModelForCausalLM, Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.lfm3 import LFM3QModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(**overrides):
    values = {
        "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "layer_types": ["conv", "full_attention", "sliding_attention"],
        "max_position_embeddings": 64, "sliding_window": 4, "conv_L_cache": 7,
        "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 3, "tie_word_embeddings": True,
    }
    values.update(overrides)
    config = Lfm3Config(**values)
    config._attn_implementation = "eager"
    return config


def shell(model):
    wrapper = LFM3QModel.__new__(LFM3QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    tree = LFM3QModel._resolve_effective_module_tree(model, None)
    wrapper.__dict__["effective_module_tree"] = tree
    wrapper.__dict__["module_tree"] = tree
    wrapper.__dict__["_module_plan"] = ModulePlan.from_tree(tree)
    wrapper.__dict__["quantize_config"] = SimpleNamespace(device="cpu", dynamic=None)
    return wrapper


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield
    torch.set_num_threads(previous)


def test_native_definition_does_not_require_checkpoint_code():
    assert not LFM3QModel.require_trust_remote_code
    model = AutoModelForCausalLM.from_config(tiny_config(), trust_remote_code=False)
    assert type(model) is Lfm3ForCausalLM


def test_awq_is_explicitly_plain_quantization_for_every_hybrid_layer():
    model = Lfm3ForCausalLM(tiny_config()).eval()
    wrapper = shell(model)
    for layer in model.model.layers:
        features = {name: torch.randn(2, 5, module.in_features)
                    for name, module in layer.named_modules() if isinstance(module, nn.Linear)}
        assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []


def save_checkpoint(path, config=None, *, legacy_gate=True):
    model = Lfm3ForCausalLM(config or tiny_config()).eval()
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if "norm.weight" in name:
                tensor.copy_(torch.linspace(0.7, 1.3, tensor.numel()).reshape_as(tensor))
    state = {key.replace(".g_proj.", ".gate_proj.") if legacy_gate else key: value
             for key, value in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state)
    tokenizer = Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5, "c": 6, "d": 7},
        unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                           bos_token="[BOS]", eos_token="[EOS]").save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {"AutoConfig": "tripwire.Config", "AutoModelForCausalLM": "tripwire.Model"}
    (path / "config.json").write_text(json.dumps(payload))
    marker = path / "EXECUTED"
    (path / "tripwire.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('checkpoint Python must not execute')\n")
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_native_load_ignores_auto_map_and_converts_gate(tmp_path, monkeypatch, trust, bypass):
    reference, marker = save_checkpoint(tmp_path, tiny_config(
        num_hidden_layers=4, layer_types=["conv", "conv", "full_attention", "sliding_attention"],
        conv_bias=True, attention_bias=True))
    if bypass is None:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    else:
        monkeypatch.setenv(BYPASS_ENV, bypass)  # Existing security policy, not a new behavior switch.
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is Lfm3Config
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    wrapper = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu",
                            dtype=torch.float32, attn_implementation="eager")
    assert type(wrapper) is LFM3QModel
    assert type(wrapper.model) is Lfm3ForCausalLM
    conv_keys = []
    for name, tensor in wrapper.model.state_dict().items():
        if ".conv.conv." not in name:
            continue
        conv_keys.append(name)
        owner = name.split(".", 3)[3].rsplit(".", 1)[0]
        spec = wrapper.module_plan().find(owner)
        assert spec is not None and spec.template == "conv.conv"
        assert "!" in spec.flags and not spec.quantize
        runtime_owner = wrapper.model.get_submodule(name.rsplit(".", 1)[0])
        assert isinstance(runtime_owner, nn.Conv1d)
        torch.testing.assert_close(getattr(runtime_owner, name.rsplit(".", 1)[1]), tensor, rtol=0, atol=0)
    assert set(conv_keys) == {f"model.layers.{i}.conv.conv.{suffix}" for i in (0, 1)
                              for suffix in ("weight", "bias")}
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(wrapper.model.state_dict()[name], tensor, rtol=0, atol=0)
    ids = torch.tensor([[4, 5, 6, 7, 4, 6]])
    with torch.inference_mode():
        torch.testing.assert_close(wrapper.model(ids, use_cache=False).logits,
                                   reference(ids, use_cache=False).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_cache_continuation_generation_and_safetensors(tmp_path, kv_heads, dtype):
    model = AutoModelForCausalLM.from_config(tiny_config(num_key_value_heads=kv_heads), dtype=dtype).eval()
    ids = torch.tensor([[0, 4, 5, 6, 7, 4, 6, 5, 7], [4, 7, 6, 5, 4, 6, 7, 4, 5]])
    mask = (ids != 0).long()
    tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        perturbed = ids.clone()
        perturbed[:, -2:] = torch.tensor([[9, 8], [10, 11]])
        causal = model(perturbed, attention_mask=mask, use_cache=False).logits
        valid = mask[:, :-2].bool()
        torch.testing.assert_close(causal[:, :-2][valid], full[:, :-2][valid], rtol=0, atol=0)
        prefix = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        cache = prefix.past_key_values
        pieces = [prefix.logits]
        for start, end in [(3, 5), (5, 8), (8, 9)]:
            result = model(ids[:, start:end], attention_mask=mask[:, :end], past_key_values=cache, use_cache=True)
            assert result.past_key_values is cache
            pieces.append(result.logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1)[mask.bool()], full[mask.bool()],
                                   rtol=tolerance, atol=tolerance)
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3,
                                   do_sample=False, eos_token_id=None)
        manual = ids.clone()
        for _ in range(3):
            growing_mask = torch.cat([mask, torch.ones(mask.shape[0], manual.shape[1] - ids.shape[1],
                                                     dtype=mask.dtype)], dim=1)
            prediction = model(manual, attention_mask=growing_mask, use_cache=False).logits[:, -1].argmax(-1)
            manual = torch.cat([manual, prediction[:, None]], dim=1)
        assert torch.equal(generated, manual)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    reloaded, info = Lfm3ForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager",
                                                  output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids, attention_mask=mask, use_cache=False).logits, full, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_hybrid_replay_matches_native_context_across_batches(dtype, explicit_positions):
    model = Lfm3ForCausalLM(tiny_config()).to(dtype).eval()
    wrapper = shell(model)
    seen = []
    handles = []

    def capture(module, args, kwargs):
        seen.append((module, args[0].clone(), kwargs.copy()))

    try:
        for layer in model.model.layers:
            handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        for ids in (torch.tensor([[0, 4, 5, 6, 7, 4], [5, 6, 7, 4, 0, 0]]),
                    torch.tensor([[4, 5, 6, 7, 4, 5, 6, 7, 4], [0, 0, 5, 6, 7, 4, 5, 6, 7]])):
            seen.clear()
            mask = (ids != 0).long()
            positions = mask.cumsum(-1).sub(1).clamp_min(0) if explicit_positions else None
            wrapper._start_outer_input_capture()
            with torch.inference_mode():
                model(ids, attention_mask=mask, position_ids=positions, use_cache=False)
            captured = wrapper.capture_first_layer_input_kwargs((), {}, torch.device("cpu"), {})
            wrapper._stop_outer_input_capture()
            native_calls = list(seen)
            for layer, inputs, native_kwargs in native_calls:
                replay = wrapper.prepare_layer_replay_kwargs(layer, [inputs], dict(captured), torch.device("cpu"))
                if layer.is_attention_layer:
                    for actual, expected in zip(replay["position_embeddings"], native_kwargs["position_embeddings"]):
                        actual, expected = torch.broadcast_tensors(actual, expected)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in ("attention_mask",):
                    if native_kwargs[name] is None:
                        assert replay.get(name) is None
                    else:
                        torch.testing.assert_close(replay[name], native_kwargs[name], rtol=0, atol=0)
                with torch.inference_mode():
                    expected = layer(inputs, **native_kwargs)
                    actual = layer(inputs, **replay)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not hasattr(model.model, "__gptqmodel_lfm3_attention_mask")
            assert not hasattr(model.model, "__gptqmodel_lfm3_position_ids")
    finally:
        wrapper._stop_outer_input_capture()
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=["mqa", "gqa", "mha"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gate_and_sandwich", [False, True])
def test_public_plain_awq_quantizes_every_linear_and_roundtrips(tmp_path, kv_heads, bias, gate_and_sandwich):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    from gptqmodel.utils.backend import BACKEND

    reference, marker = save_checkpoint(tmp_path, tiny_config(num_key_value_heads=kv_heads,
                                                             conv_bias=bias, attention_bias=bias,
                                                             use_attention_gate=gate_and_sandwich,
                                                             use_sandwich_norm=gate_and_sandwich))
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                            backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager")
    excluded = {key: value.to(torch.float16).clone() for key, value in reference.state_dict().items()
                if any(name in key for name in (".g_proj.", ".conv.conv.", "norm.weight"))}
    rows = [torch.tensor([[4, 5, 6, 7, 4, 5, 7, 6, 4]]), torch.tensor([[5, 7, 4, 6, 5, 4, 6, 7, 5]])]
    calibration = [{"input_ids": row, "attention_mask": torch.ones_like(row)} for row in rows]
    wrapper.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    quantized = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected = {f"model.layers.{i}.{name}" for i in range(3) for name in (
        ["conv.in_proj", "conv.out_proj"] if i == 0 else
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"])}
    expected |= {f"model.layers.{i}.feed_forward.w{j}" for i in range(3) for j in (1, 2, 3)}
    assert quantized == expected
    for key, tensor in excluded.items():
        torch.testing.assert_close(wrapper.model.state_dict()[key], tensor, rtol=0, atol=0)
    with torch.inference_mode():
        logits = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(logits).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16,
                           attn_implementation="eager")
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**calibration[0], use_cache=False).logits,
                                   logits, rtol=0, atol=0)
    for key, tensor in wrapper.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], tensor, rtol=0, atol=0)
    assert not marker.exists()


def test_public_native_load_failure_then_retry_does_not_execute_checkpoint_code(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    reference, marker = save_checkpoint(tmp_path)
    monkeypatch.setenv(BYPASS_ENV, "1")
    checkpoint = tmp_path / "model.safetensors"
    state = load_file(checkpoint)
    key = "model.layers.1.self_attn.gate_proj.weight"
    broken = {**state, key: state[key][:1].clone()}
    save_file(broken, checkpoint, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="ignore_mismatched_sizes"):
        GPTQModel.load(str(tmp_path), trust_remote_code=True, device="cpu", dtype=torch.float32)
    assert not marker.exists()
    save_file(state, checkpoint, metadata={"format": "pt"})
    retried = GPTQModel.load(str(tmp_path), trust_remote_code=True, device="cpu", dtype=torch.float32)
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(retried.model.state_dict()[key], tensor, rtol=0, atol=0)
    assert not marker.exists()
