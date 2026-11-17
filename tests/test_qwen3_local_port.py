# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3 and the published Qwen3-backed Marin release."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_5_text import Qwen3_5TextQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
from gptqmodel.utils.hf import load_model_config


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/qwen3/fixtures"


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def asset(family, name):
    return gzip.decompress((FIXTURES / f"{family}-{name}.gz").read_bytes())


def write_assets(path, family):
    path.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        (path / name).write_bytes(asset(family, name))
    if family == "marin":
        (path / "special_tokens_map.json").write_bytes(asset(family, "special_tokens_map.json"))


def hostile_metadata(path):
    marker = path / "EXECUTED"
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = dict.fromkeys(["AutoConfig", "AutoModelForCausalLM", "AutoTokenizer"], "untrusted.Untrusted")
    (path / "config.json").write_text(json.dumps(payload))
    payload = json.loads((path / "tokenizer_config.json").read_text())
    payload["auto_map"] = {"AutoTokenizer": ["untrusted.Untrusted", "untrusted.Untrusted"]}
    (path / "tokenizer_config.json").write_text(json.dumps(payload))
    (path / "untrusted.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    (path / "custom_generate").mkdir(exist_ok=True)
    (path / "custom_generate/generate.py").write_text((path / "untrusted.py").read_text())
    return marker


def security_env(monkeypatch, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)


def source_fixture(path, family, **overrides):
    write_assets(path, family)
    source = json.loads(asset(family, "config.json"))
    config = tiny_config(
        vocab_size=source["vocab_size"],
        bos_token_id=source.get("bos_token_id"),
        eos_token_id=source["eos_token_id"],
        **overrides,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(199)
        model = Qwen3ForCausalLM(config).eval()
    model.generation_config = (
        GenerationConfig.from_dict(json.loads(asset(family, "generation_config.json")))
        if family == "qwen3"
        else GenerationConfig.from_model_config(Qwen3Config.from_dict(source))
    )
    model.save_pretrained(path, safe_serialization=True)
    return model


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    config = Qwen3Config(**values)
    config._attn_implementation = "eager"
    return config


def test_definition_owns_exact_native_qwen3():
    assert isinstance(Qwen3QModel.loader, LocalModelLoader)
    model = Qwen3QModel.loader.from_config(tiny_config())
    assert type(model) is Qwen3ForCausalLM
    assert type(model.config) is Qwen3Config
    assert not Qwen3QModel.require_trust_remote_code
    assert Qwen3_5TextQModel.loader is AutoModelForCausalLM
    assert AutoConfig.for_model("qwen3").__class__ is Qwen3Config
    assert type(AutoModelForCausalLM.from_config(tiny_config())) is Qwen3ForCausalLM


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_raw_marin_metadata_rejected_before_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = tiny_config().to_dict()
    config.update(
        model_type="marin",
        architectures=["Qwen3ForCausalLM"],
        auto_map={"AutoConfig": "configuration_untrusted.UntrustedConfig"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    marker = tmp_path / "EXECUTED"
    (tmp_path / "configuration_untrusted.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    with pytest.raises(ValueError, match="Marin.*model_type=qwen3"):
        load_model_config(tmp_path, trust_remote_code=trust)
    assert not marker.exists()


@pytest.mark.parametrize("family", ["marin", "qwen3"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_native_selection_tokenizer_generation_and_reload(tmp_path, monkeypatch, family, trust, bypass):
    security_env(monkeypatch, bypass)
    source = source_fixture(tmp_path / "source", family)
    path = tmp_path / "source"
    safe_tokenizer = AutoTokenizer.from_pretrained(path, config=source.config, trust_remote_code=False)
    ids = safe_tokenizer("Marin and Qwen: café 你好!", return_tensors="pt").input_ids
    expected_ids = {
        "marin": [128000, 12331, 258, 323, 1229, 17378, 25, 53050, 118195, 53901, 0],
        "qwen3": [12061, 258, 323, 1207, 16948, 25, 51950, 220, 108386, 0],
    }
    assert ids[0].tolist() == expected_ids[family]
    marker = hostile_metadata(path)
    loaded = GPTQModel.load(
        str(path),
        device="cpu",
        backend=BACKEND.TORCH,
        dtype=torch.float32,
        trust_remote_code=trust,
        attn_implementation="eager",
    )
    assert type(loaded.model) is Qwen3ForCausalLM
    assert type(loaded.model.config) is Qwen3Config
    assert loaded.tokenizer.encode("Marin and Qwen: café 你好!") == ids[0].tolist()
    assert loaded.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True) == "Marin and Qwen: café 你好!"
    if family == "qwen3":
        messages = [{"role": "user", "content": "Hello 你好"}]
        assert loaded.tokenizer.apply_chat_template(messages) == safe_tokenizer.apply_chat_template(messages)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, rtol=0, atol=0)
        for beams in [1, 3]:
            options = {"max_new_tokens": 3, "do_sample": False, "num_beams": beams}
            assert torch.equal(loaded.model.generate(ids, **options), source.generate(ids, **options))
    for cycle in range(2):
        target = tmp_path / f"dense-{cycle}"
        loaded.model.save_pretrained(target, safe_serialization=True)
        loaded.tokenizer.save_pretrained(target)
        loaded = GPTQModel.load(
            str(target),
            device="cpu",
            backend=BACKEND.TORCH,
            dtype=torch.float32,
            trust_remote_code=trust,
            attn_implementation="eager",
        )
        assert type(loaded.model) is Qwen3ForCausalLM
        for name in [
            "bos_token_id",
            "eos_token_id",
            "begin_suppress_tokens",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
        ]:
            assert getattr(loaded.model.generation_config, name) == getattr(source.generation_config, name)
        assert loaded.tokenizer.encode("Marin and Qwen: café 你好!") == ids[0].tolist()
        for key, value in source.state_dict().items():
            torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
        assert list(target.glob("*.safetensors"))
        assert not list(target.glob("*.bin"))
    assert not marker.exists()


def test_pinned_assets_integrity():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for family, (repo, revision) in manifest["releases"].items():
        for path in FIXTURES.glob(f"{family}-*.gz"):
            name = path.name[len(family) + 1 : -3]
            if name == "headers.json":
                continue
            expected = manifest["assets"][f"https://huggingface.co/{repo}/resolve/{revision}/{name}"]
            raw = gzip.decompress(path.read_bytes())
            assert len(raw) == expected["bytes"]
            assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_raw_marin_public_fail_repair_retry_same_checkpoint(tmp_path, monkeypatch, trust, bypass):
    security_env(monkeypatch, bypass)
    source = source_fixture(tmp_path, "marin")
    marker = hostile_metadata(tmp_path)
    config_path = tmp_path / "config.json"
    valid = config_path.read_bytes()
    payload = json.loads(valid)
    payload["model_type"] = "marin"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Marin.*model_type=qwen3"):
        GPTQModel.load(str(tmp_path), device="cpu", backend=BACKEND.TORCH, trust_remote_code=trust)
    assert not marker.exists()
    config_path.write_bytes(valid)
    loaded = GPTQModel.load(
        str(tmp_path),
        device="cpu",
        backend=BACKEND.TORCH,
        dtype=torch.float32,
        trust_remote_code=trust,
        attn_implementation="eager",
    )
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    ids = loaded.tokenizer("retry café", return_tensors="pt").input_ids
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, source(ids).logits, rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("family", ["marin", "qwen3"])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("kv_heads,bias", [(4, False), (2, True), (1, False)])
def test_public_plain_awq_pack_infer_two_exact_reloads(tmp_path, family, offload, kv_heads, bias):
    source = source_fixture(tmp_path / "source", family, num_key_value_heads=kv_heads, attention_bias=bias)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    loaded = GPTQModel.load(
        str(tmp_path / "source"),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    assert loaded.awq_get_modules_for_scaling(loaded.model.model.layers[0], {}, {}) == []
    dense = {
        key: value.detach().to(torch.float16).clone()
        for key, value in source.state_dict().items()
        if not any(
            f".{projection}." in key
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    }
    rows = [
        loaded.tokenizer(text, return_tensors="pt")
        for text in ["Marin and Qwen models learn café 你好.", "A different sequence: 42, seven, moon!"]
    ]
    loaded.quantize(CalibrationConfig(dataset=rows, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    assert sum(isinstance(m, AwqTorchLinear) for m in loaded.model.modules()) == 14
    for key, value in dense.items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    packed = {key: value.detach().clone() for key, value in loaded.model.state_dict().items()}
    assert sum(key.endswith("qweight") for key in packed) == 14
    assert sum(key.endswith("qzeros") for key in packed) == 14
    with torch.inference_mode():
        expected = loaded.model(**rows[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
    for cycle in range(2):
        target = tmp_path / f"quantized-{cycle}"
        loaded.save_quantized(str(target))
        loaded = GPTQModel.load(
            str(target), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
        )
        assert type(loaded.model) is Qwen3ForCausalLM
        assert set(loaded.model.state_dict()) == set(packed)
        for key, value in packed.items():
            actual = loaded.model.state_dict()[key]
            assert actual.dtype == value.dtype, key
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(**rows[0], use_cache=False).logits, expected, rtol=0, atol=0)
        assert list(target.glob("*.safetensors"))
        assert not list(target.glob("*.bin"))


def independent_decoder(model, ids, positions):
    """Explicit float64 equations; never call native norm/attention/RoPE helpers."""
    state = {key: value.detach().double() for key, value in model.state_dict().items()}
    config = model.config
    head_dim = config.head_dim
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads

    def linear(x, path):
        bias = state.get(path + ".bias", 0)
        return x @ state[path + ".weight"].T + bias

    def norm(x, path):
        return x * (x.square().mean(-1, keepdim=True) + config.rms_norm_eps).rsqrt() * state[path + ".weight"]

    rope = config.rope_parameters
    freq = rope["rope_theta"] ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    if rope["rope_type"] == "llama3":
        wavelength = 2 * torch.pi / freq
        old = rope["original_max_position_embeddings"]
        low, high = rope["low_freq_factor"], rope["high_freq_factor"]
        smooth = (old / wavelength - low) / (high - low)
        middle = (1 - smooth) * freq / rope["factor"] + smooth * freq
        freq = torch.where(
            wavelength > old / low, freq / rope["factor"], torch.where(wavelength < old / high, freq, middle)
        )
    angles = positions.double()[..., None] * freq
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)

    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return x * angles.cos() + torch.cat([-right, left], dim=-1) * angles.sin()

    x = state["model.embed_tokens.weight"][ids]
    length = ids.shape[1]
    row = torch.arange(length)
    for index in range(config.num_hidden_layers):
        root = f"model.layers.{index}."
        y = norm(x, root + "input_layernorm")
        queries = linear(y, root + "self_attn.q_proj").view(*ids.shape, heads, head_dim)
        keys = linear(y, root + "self_attn.k_proj").view(*ids.shape, kv_heads, head_dim)
        values = linear(y, root + "self_attn.v_proj").view(*ids.shape, kv_heads, head_dim).transpose(1, 2)
        queries = rotate(norm(queries, root + "self_attn.q_norm").transpose(1, 2))
        keys = rotate(norm(keys, root + "self_attn.k_norm").transpose(1, 2))
        keys = keys.repeat_interleave(heads // kv_heads, dim=1)
        values = values.repeat_interleave(heads // kv_heads, dim=1)
        scores = queries @ keys.transpose(-1, -2) / head_dim**0.5
        permitted = row[None, :] <= row[:, None]
        if config.layer_types[index] == "sliding_attention":
            permitted &= row[None, :] > row[:, None] - config.sliding_window
        scores = scores.masked_fill(~permitted, -torch.inf).softmax(-1)
        context = (scores @ values).transpose(1, 2).reshape(*ids.shape, heads * head_dim)
        x = x + linear(context, root + "self_attn.o_proj")
        y = norm(x, root + "post_attention_layernorm")
        gate = linear(y, root + "mlp.gate_proj")
        x = x + linear(gate.sigmoid() * gate * linear(y, root + "mlp.up_proj"), root + "mlp.down_proj")
    return linear(norm(x, "model.norm"), "lm_head")


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("llama3_rope", [False, True])
@pytest.mark.parametrize("sliding", [False, True])
def test_independent_qk_norm_rotary_decoder_oracle(kv_heads, bias, llama3_rope, sliding):
    options = {
        "num_key_value_heads": kv_heads,
        "attention_bias": bias,
        "head_dim": 10,
        "use_sliding_window": sliding,
        "sliding_window": 3,
        "max_window_layers": 1,
    }
    if llama3_rope:
        options["rope_parameters"] = {
            "rope_type": "llama3",
            "rope_theta": 500000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 16,
        }
    model = Qwen3QModel.loader.from_config(tiny_config(**options)).eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(40)
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.linspace(0.3, 1.7, parameter.numel()).reshape_as(parameter))
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.12)
    ids = torch.tensor([[5, 9, 4, 21, 7], [19, 4, 10, 8, 17]])
    positions = torch.arange(65530, 65535).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        actual = model(ids, position_ids=positions, use_cache=False).logits
        expected = independent_decoder(model, ids, positions)
    # Native RoPE intentionally computes positions in FP32. At 65k its
    # trigonometric argument rounds; the independent double oracle bounds it.
    torch.testing.assert_close(actual.double(), expected, rtol=4e-3, atol=6e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("sliding", [False, True])
def test_native_cache_continuation_assignment_and_safetensors_cycles(tmp_path, dtype, implementation, sliding):
    config = tiny_config(use_sliding_window=sliding, sliding_window=3, max_window_layers=1)
    config._attn_implementation = implementation
    model = Qwen3QModel.loader.from_config(config, dtype=dtype).eval()
    ids = torch.tensor([[5, 9, 4, 21, 7, 6], [19, 4, 10, 8, 17, 20]])
    tolerance = {torch.float32: 2e-6, torch.float16: 4e-4, torch.bfloat16: 4e-3}[dtype]
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        cache = None
        offset = 0
        for chunk in [1, 4, 1]:
            output = model(ids[:, offset : offset + chunk], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            torch.testing.assert_close(output.logits, full[:, offset : offset + chunk], rtol=0, atol=tolerance)
            offset += chunk
        changed = ids.clone()
        changed[:, -1] = 25
        torch.testing.assert_close(model(changed).logits[:, :-1], full[:, :-1], rtol=0, atol=0)
    original = copy.deepcopy(model.state_dict())
    for assign in [False, True]:
        fresh = Qwen3QModel.loader.from_config(copy.deepcopy(config), dtype=dtype).eval()
        fresh.load_state_dict(original, assign=assign)
        with torch.inference_mode():
            torch.testing.assert_close(fresh(ids, use_cache=False).logits, full, rtol=0, atol=0)
    for cycle in range(2):
        target = tmp_path / str(cycle)
        model.save_pretrained(target, safe_serialization=True)
        model = Qwen3QModel.loader.from_pretrained(target, dtype=dtype, attn_implementation=implementation).eval()
        assert type(model) is Qwen3ForCausalLM
        for key, value in original.items():
            assert model.state_dict()[key].dtype == value.dtype
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids, use_cache=False).logits, full, rtol=0, atol=0)


def test_subfolder_selection_and_corrupt_safetensors_repair_retry(tmp_path):
    from safetensors import SafetensorError

    path = tmp_path / "nested"
    model = source_fixture(path, "marin")
    # The root is a decoy and must not participate in config dispatch.
    (tmp_path / "config.json").write_text('{"model_type": "marin"}')
    tensor_path = path / "model.safetensors"
    original = tensor_path.read_bytes()
    tensor_path.write_bytes(b"broken header")
    with pytest.raises((OSError, ValueError, SafetensorError)):
        GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu", backend=BACKEND.TORCH)
    tensor_path.write_bytes(original)
    loaded = GPTQModel.load(
        str(tmp_path),
        subfolder="nested",
        device="cpu",
        backend=BACKEND.TORCH,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    assert type(loaded.model) is Qwen3ForCausalLM
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
