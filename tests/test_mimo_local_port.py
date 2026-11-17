# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MiMo's real Qwen2 extension must never enter checkpoint Python."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import SafetensorError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.mimo import MimoQModel
from gptqmodel.models.ports.mimo.configuration_mimo import MiMoConfig
from gptqmodel.models.ports.mimo.modeling_mimo import MiMoForCausalLM
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config


ASSETS = Path(__file__).parents[1] / "gptqmodel/models/ports/mimo/fixtures"


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    values = {
        "vocab_size": 48, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_nextn_predict_layers": 1,
        "max_position_embeddings": 64, "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0,
        "rope_theta": 640000, "rms_norm_eps": 1e-5,
    }
    values.update(kwargs)
    config = MiMoConfig(**values)
    config._attn_implementation = "eager"
    return config


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return MiMoForCausalLM(tiny_config(**kwargs)).eval()


def checkpoint(path, **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    model = tiny_model(**kwargs)
    model.save_pretrained(path)
    tokenizer = Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5, "c": 6, "d": 7},
        unk_token="[UNK]",
    ))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
    ).save_pretrained(path)
    marker = path / "executed"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = {
        "AutoConfig": "evil.Config", "AutoModel": "evil.Model", "AutoModelForCausalLM": "evil.Model",
    }
    (path / "config.json").write_text(json.dumps(payload))
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_mimo_config_selection_ignores_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = tmp_path / "executed"
    (tmp_path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "mimo", "hidden_size": 32, "intermediate_size": 64,
        "vocab_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_nextn_predict_layers": 1,
        "auto_map": {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"},
    }))
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config).__module__.startswith("gptqmodel.models.ports.mimo.")
    assert config.num_nextn_predict_layers == 1
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_load_local_model_and_tokenizer(tmp_path, monkeypatch, trust, bypass):
    expected, marker = checkpoint(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                           attn_implementation="eager")
    assert isinstance(loaded, MimoQModel)
    assert type(loaded.model) is MiMoForCausalLM
    assert type(loaded.model.config) is MiMoConfig
    assert len(loaded.model.model.mtp_layers) == 1
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, expected(ids).logits, rtol=0, atol=0)
    assert set(loaded.model.state_dict()) == set(expected.state_dict())
    assert loaded.module_plan().find("self_attn.q_proj").quantize
    assert loaded.tokenizer is not None
    assert not marker.exists()


def _norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.double()


def _linear(x, state, prefix):
    bias = state.get(prefix + ".bias")
    return F.linear(x, state[prefix + ".weight"].double(), None if bias is None else bias.double())


def _attention(x, state, prefix, config, mask, positions):
    batch, seq, _ = x.shape
    heads, kv = config.num_attention_heads, config.num_key_value_heads
    dim = getattr(config, "head_dim", config.hidden_size // heads)
    q = _linear(x, state, prefix + ".q_proj").view(batch, seq, heads, dim).transpose(1, 2)
    k = _linear(x, state, prefix + ".k_proj").view(batch, seq, kv, dim).transpose(1, 2)
    v = _linear(x, state, prefix + ".v_proj").view(batch, seq, kv, dim).transpose(1, 2)
    theta = config.rope_parameters["rope_theta"]
    freq = positions.double().unsqueeze(-1) / theta ** (torch.arange(0, dim, 2).double() / dim)
    angles = torch.cat([freq, freq], -1).unsqueeze(1)

    def rotate(t):
        half = dim // 2
        return t * angles.cos() + torch.cat([-t[..., half:], t[..., :half]], -1) * angles.sin()

    q, k = rotate(q), rotate(k)
    k, v = k.repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
    scores = q @ k.transpose(-1, -2) / dim**0.5
    allowed = torch.ones(seq, seq, dtype=torch.bool).tril()[None, None] & mask[:, None, None, :].bool()
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, seq, heads * dim)
    return _linear(y, state, prefix + ".o_proj")


def _mlp(x, state, prefix):
    return _linear(F.silu(_linear(x, state, prefix + ".gate_proj")) * _linear(x, state, prefix + ".up_proj"),
                   state, prefix + ".down_proj")


def independent_logits(model, ids, mask, positions):
    """Double precision equations, with no native Qwen2 forward helpers."""
    state, config = model.state_dict(), model.config
    x = F.embedding(ids, state["model.embed_tokens.weight"].double())
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        y = _norm(x, state[prefix + ".input_layernorm.weight"], config.rms_norm_eps)
        x = x + _attention(y, state, prefix + ".self_attn", config, mask, positions)
        y = _norm(x, state[prefix + ".post_attention_layernorm.weight"], config.rms_norm_eps)
        x = x + _mlp(y, state, prefix + ".mlp")
    return _linear(_norm(x, state["model.norm.weight"], config.rms_norm_eps), state, "lm_head")


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_forward_matches_independent_equations(kv, dtype):
    model = tiny_model(num_key_value_heads=kv).to(dtype)
    # Non-unit gains and nonzero QKV biases prevent vacuous norm/bias parity.
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "layernorm.weight" in name or name == "model.norm.weight":
                value.copy_(torch.linspace(0.7, 1.3, value.numel()).reshape_as(value))
            if name.endswith(".bias"):
                value.copy_(torch.linspace(-0.1, 0.1, value.numel()))
    ids = torch.tensor([[4, 5, 6, 7, 8], [9, 10, 11, 12, 13]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    positions = torch.arange(ids.shape[1]).expand_as(ids) + 3
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        expected = independent_logits(model, ids, mask, positions)
    # A fully masked padding query has implementation-dependent softmax; real
    # tokens, including causal queries after that padding, must agree.
    atol = {torch.float32: 2e-7, torch.float16: 8e-4, torch.bfloat16: 7e-3}[dtype]
    torch.testing.assert_close(actual[mask.bool()].double(), expected[mask.bool()], atol=atol, rtol=atol)


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("mtp", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_qwen_parity_generation_cache_and_mtp_isolation(implementation, kv, mtp, dtype):
    model = tiny_model(num_key_value_heads=kv, num_nextn_predict_layers=mtp).to(dtype)
    model.set_attn_implementation(implementation)
    config = Qwen2Config.from_dict(model.config.to_dict())
    config._attn_implementation = implementation
    reference = Qwen2ForCausalLM(config).to(dtype).eval()
    native_state = {k: v for k, v in model.state_dict().items() if not k.startswith("model.mtp_layers.")}
    reference.load_state_dict(native_state, strict=True)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    mask[1, 0] = 0
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=True)
        torch.testing.assert_close(full.logits, reference(ids, attention_mask=mask).logits, rtol=0, atol=0)
        cache = model(ids[:, :1], attention_mask=mask[:, :1], use_cache=True).past_key_values
        for start, end in [(1, 5), (5, 6)]:
            out = model(ids[:, start:end], attention_mask=mask[:, :end], past_key_values=cache, use_cache=True)
            tolerance = {torch.float32: 1e-6, torch.float16: 5e-4, torch.bfloat16: 4e-3}[dtype]
            torch.testing.assert_close(out.logits, full.logits[:, start:end], rtol=tolerance, atol=tolerance)
            cache = out.past_key_values
        assert cache.get_seq_length() == ids.shape[1]
        changed = ids.clone()
        changed[:, -1] = 17
        valid = mask[:, :-1].bool()
        torch.testing.assert_close(model(changed, attention_mask=mask).logits[:, :-1][valid],
                                   full.logits[:, :-1][valid], rtol=0, atol=0)
        for beams in [1, 2]:
            assert torch.equal(
                model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=beams),
                reference.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=beams),
            )
        for module in model.model.mtp_layers:
            for parameter in module.parameters():
                parameter.fill_(123)
        torch.testing.assert_close(model(ids, attention_mask=mask).logits, full.logits, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_safetensors_roundtrip_and_failed_load_retry(tmp_path, dtype, tied):
    expected = tiny_model(tie_word_embeddings=tied, num_nextn_predict_layers=2).to(dtype)
    expected.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    tensor_file = tmp_path / "model.safetensors"
    contents = tensor_file.read_bytes()
    tensor_file.write_bytes(b"corrupt safetensors")
    with pytest.raises(SafetensorError, match="header|Header|safetensor"):
        MimoQModel.loader.from_pretrained(tmp_path, dtype=dtype)
    tensor_file.write_bytes(contents)
    loaded = MimoQModel.loader.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager").eval()
    assert set(loaded.state_dict()) == set(expected.state_dict())
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tied
    ids = torch.tensor([[4, 5, 6, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, expected(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_invalid_mtp_count_fails_then_valid_config_retries(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        tiny_config(num_nextn_predict_layers=value)
    assert len(tiny_model(num_nextn_predict_layers=1).model.mtp_layers) == 1


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_mtp_equations_rotary_api_and_separate_cache(kv):
    model = tiny_model(num_key_value_heads=kv)
    mtp = model.model.mtp_layers[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        embeds, hidden = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    state, config = mtp.state_dict(), model.config
    x = torch.cat([_norm(hidden.double(), state["hidden_layernorm.weight"], config.rms_norm_eps),
                   _norm(embeds.double(), state["token_layernorm.weight"], config.rms_norm_eps)], -1)
    x = _linear(x, state, "input_proj")
    positions = torch.arange(5).expand(2, -1)
    x = x + _attention(_norm(x, state["input_layernorm.weight"], config.rms_norm_eps),
                       state, "self_attn", config, torch.ones(2, 5), positions)
    x = x + _mlp(_norm(x, state["post_attention_layernorm.weight"], config.rms_norm_eps), state, "mlp")
    expected = _norm(x, state["final_layernorm.weight"], config.rms_norm_eps)
    mask = torch.full((2, 1, 5, 5), torch.finfo(torch.float32).min).triu(1)
    with torch.inference_mode():
        rotary = model.model.rotary_emb(embeds, positions)
        with pytest.raises(ValueError, match="position_embeddings"):
            mtp(embeds, hidden, mask)
        full = mtp(embeds, hidden, mask, position_embedding=rotary)
        torch.testing.assert_close(full.double(), expected, rtol=3e-6, atol=6e-7)
        torch.testing.assert_close(mtp(embeds, hidden, mask, position_embeddings=rotary), full, rtol=0, atol=0)
        cache = DynamicCache(config=config)
        for start, end in [(0, 1), (1, 5)]:
            out = mtp(embeds[:, start:end], hidden[:, start:end], mask[:, :, start:end, :end],
                      position_embeddings=tuple(v[:, start:end] for v in rotary),
                      past_key_values=cache, use_cache=True)
            torch.testing.assert_close(out, full[:, start:end], atol=3e-7, rtol=3e-6)
        assert cache.get_seq_length() == 5


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("fallback", [None, {}])
def test_public_plain_awq_packs_every_decoder_and_preserves_dense_mtp(tmp_path, kv, fallback):
    source = tmp_path / "source"
    expected, marker = checkpoint(source, num_key_value_heads=kv)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=fallback, offload_to_disk=False)
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert model.awq_get_modules_for_scaling(model.model.model.layers[0], {}, {}) == []
    dense = {k: v.clone() for k, v in model.model.state_dict().items()
             if not k.startswith("model.layers.") or "layernorm" in k}
    original_weights = {k: v.clone() for k, v in model.model.state_dict().items() if k.endswith(".weight")}
    calibration = [
        {"input_ids": torch.tensor([[4, 5, 6, 7, 4, 5, 6, 7]]),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.tensor([[7, 5, 4, 6, 7, 6, 4, 5]]),
         "attention_mask": torch.ones(1, 8, dtype=torch.long)},
    ]
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == {f"model.layers.{i}.{name}" for i in range(2) for name in [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]}
    assert len(packed) == 14
    assert all(module.qweight.dtype == torch.int32 for module in model.model.modules()
               if isinstance(module, AwqTorchLinear))
    for key, value in dense.items():
        torch.testing.assert_close(model.model.state_dict()[key], value, rtol=0, atol=0)
    assert any(key + ".qweight" in model.model.state_dict() for key in packed)
    assert all(name + ".weight" not in model.model.state_dict() for name in packed)
    # Every original decoder weight became packed integer state; the original
    # dense copies are held by this test only, never by the inference module.
    assert set(packed) == {k.removesuffix(".weight") for k in original_weights
                           if k.startswith("model.layers.") and "layernorm" not in k}
    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits
        generated = model.model.generate(**calibration[0], max_new_tokens=3, do_sample=False)
    assert torch.isfinite(logits).all()
    destination = tmp_path / "quantized"
    model.save(str(destination))
    assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
    reloaded = GPTQModel.load(str(destination), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
    assert type(reloaded.model) is MiMoForCausalLM
    assert len(reloaded.model.model.mtp_layers) == 1
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(reloaded.model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in dense.items():
        torch.testing.assert_close(reloaded.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, logits, rtol=0, atol=0)
        assert torch.equal(reloaded.model.generate(**calibration[0], max_new_tokens=3, do_sample=False), generated)
    assert not marker.exists()
    del expected, model, reloaded, original_weights


def test_public_subfolder_selection_and_corrupt_config_repair_retry(tmp_path):
    path = tmp_path / "nested"
    expected, marker = checkpoint(path)
    config_path = path / "config.json"
    contents = config_path.read_text()
    config_path.write_text("{invalid")
    with pytest.raises(OSError, match="JSON"):
        GPTQModel.load(str(tmp_path), subfolder="nested", dtype=torch.float32, device="cpu")
    assert not marker.exists()
    config_path.write_text(contents)
    loaded = GPTQModel.load(str(tmp_path), subfolder="nested", dtype=torch.float32, device="cpu",
                           attn_implementation="eager", trust_remote_code=True)
    assert type(loaded.model) is MiMoForCausalLM
    with torch.inference_mode():
        ids = torch.tensor([[4, 5, 6]])
        torch.testing.assert_close(loaded.model(ids).logits, expected(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_optional_native_sliding_window_layout_cache_continuation(implementation):
    model = tiny_model(use_sliding_window=True, sliding_window=3, max_window_layers=1)
    model.set_attn_implementation(implementation)
    assert model.config.layer_types == ["full_attention", "sliding_attention"]
    assert model.model.layers[1].self_attn.sliding_window == 3
    config = Qwen2Config.from_dict(model.config.to_dict())
    config._attn_implementation = implementation
    reference = Qwen2ForCausalLM(config).eval()
    reference.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("model.mtp_layers.")})
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    with torch.inference_mode():
        full = model(ids).logits
        torch.testing.assert_close(full, reference(ids).logits, rtol=0, atol=0)
        first = model(ids[:, :1], use_cache=True)
        continued = model(ids[:, 1:5], past_key_values=first.past_key_values, use_cache=True)
        torch.testing.assert_close(continued.logits, full[:, 1:5], rtol=1e-5, atol=1e-7)
        last = model(ids[:, 5:], past_key_values=continued.past_key_values, use_cache=True)
        torch.testing.assert_close(last.logits, full[:, 5:], rtol=1e-5, atol=1e-7)
