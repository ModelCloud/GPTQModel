# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native Unlimited-OCR task, legacy conversion and public lifecycle qualification."""

import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from transformers import (
    UnlimitedOcrConfig,
    UnlimitedOcrForConditionalGeneration,
)

from gptqmodel.models.definitions.unlimited_ocr import UnlimitedOCRQModel
from gptqmodel.utils.hf import load_model_config


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/unlimited_ocr/fixtures"
UNUSED_PATCH_NAME = "model.vision_model.embeddings.patch_embedding.weight"


def asset(name):
    path = FIXTURES / name
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())


def legacy_key(name):
    """Independent inverse of the inspected Baidu names (not HF conversion helpers)."""
    name = name.replace("model.vision_tower.sam_encoder.", "model.sam_model.")
    name = name.replace("model.vision_tower.vision_encoder.", "model.vision_model.")
    name = name.replace("model.language_model.", "model.")
    name = name.replace("model.view_separator", "model.view_seperator")
    name = name.replace("model.multi_modal_projector.", "model.projector.layers.")
    if name.startswith("model.sam_model."):
        name = re.sub(r"layers\.(\d+)\.layer_norm([12])\.", r"blocks.\1.norm\2.", name)
        name = re.sub(r"layers\.(\d+)\.(attn|mlp)\.", r"blocks.\1.\2.", name)
        for a, b in [("patch_embed.projection.", "patch_embed.proj."),
                     ("neck.conv1.", "neck.0."), ("neck.layer_norm1.", "neck.1."),
                     ("neck.conv2.", "neck.2."), ("neck.layer_norm2.", "neck.3."),
                     ("proj.conv1.", "net_2."), ("proj.conv2.", "net_3.")]:
            name = name.replace(a, b)
    return name.replace("model.vision_model.encoder.", "model.vision_model.transformer.")


def source_state(model):
    from defuser.modeling.update_module import update_module

    update_module(model, cleanup_original=False)
    output = {}
    for name, value in model.state_dict().items():
        key = legacy_key(name)
        if key.startswith("model.vision_model.") and re.search(r"[qkv]_proj\.(weight|bias)$", key):
            if ".q_proj." not in key:
                continue
            output[key.replace(".q_proj.", ".qkv_proj.")] = torch.cat([
                model.state_dict()[name.replace(".q_proj.", f".{p}_proj.")] for p in "qkv"
            ]).clone()
        else:
            output[key] = value.detach().clone()
    encoder = model.config.vision_config.encoder_config
    shape = (encoder.hidden_size, encoder.num_channels, encoder.patch_size, encoder.patch_size)
    output[UNUSED_PATCH_NAME] = torch.arange(torch.tensor(shape).prod()).float().reshape(shape) / 100
    return output


def source_fixture(path, *, legacy=False, layout="mixed", kv_heads=4, bias=False, released_vocab=False):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(147)
    config = tiny_config(kv_heads, bias, layout=layout)
    if released_vocab:
        config.text_config.vocab_size = 129280
        config.text_config.pad_token_id = None
        config.image_token_id = 128815
    model = UnlimitedOcrForConditionalGeneration(config).eval()
    with torch.no_grad():
        model.model.image_newline.normal_(std=0.05)
        model.model.view_separator.normal_(std=0.05)
    if legacy:
        state = source_state(model)
        save_file(state, path / "model.safetensors", metadata={"format": "pt"})
        payload = model.config.to_dict()
        payload["language_config"] = payload.pop("text_config")
        payload["model_type"] = "unlimited-ocr"
        (path / "config.json").write_text(json.dumps(payload))
    else:
        model.save_pretrained(path)
    if released_vocab:
        (path / "generation_config.json").write_bytes(asset("generation_config.json"))
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        (path / name).write_bytes(asset(name))
    from transformers import UnlimitedOcrImageProcessor

    UnlimitedOcrImageProcessor(
        size={"height": 32, "width": 32}, tile_size=32, min_patches=2, max_patches=4,
    ).save_pretrained(path)
    return model


def hostile(path):
    marker = path / "EXECUTED"
    (path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint Python')\n"
    )
    for name in ["config.json", "tokenizer_config.json", "preprocessor_config.json"]:
        payload = json.loads((path / name).read_bytes())
        payload["auto_map"] = {
            "AutoConfig": "evil.Config", "AutoModel": "evil.Model",
            "AutoModelForCausalLM": "evil.Model", "AutoTokenizer": ["evil.Tokenizer", None],
            "AutoProcessor": "evil.Processor", "AutoImageProcessor": "evil.ImageProcessor",
        }
        (path / name).write_text(json.dumps(payload))
    (path / "processor_config.json").write_text(json.dumps({
        "processor_class": "EvilProcessor", "auto_map": {"AutoProcessor": "evil.Processor"},
    }))
    return marker


def tiny_config(kv_heads=4, bias=False, dtype=torch.float32, layout="mixed"):
    config = UnlimitedOcrConfig(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": kv_heads, "attention_bias": bias,
            "mlp_bias": bias, "n_routed_experts": 4, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "moe_intermediate_size": 16,
            "sliding_window": 4, "pad_token_id": 2,
            "mlp_layer_types": {"mixed": ["dense", "sparse"], "dense": ["dense"] * 2,
                                "sparse": ["sparse"] * 2}[layout],
        },
        vision_config={
            "sam_config": {"hidden_size": 8, "output_channels": 4,
                           "num_hidden_layers": 1, "num_attention_heads": 2,
                           "image_size": 32, "patch_size": 4, "window_size": 2,
                           "global_attn_indexes": [0], "mlp_dim": 16,
                           "downsample_channels": [8, 8], "initializer_range": 0.02},
            "encoder_config": {"hidden_size": 8, "intermediate_size": 16,
                               "num_hidden_layers": 1, "num_attention_heads": 2,
                               "image_size": 8, "patch_size": 1},
        }, image_token_id=3, dtype=dtype,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("model_type", ["unlimited_ocr", "unlimited-ocr"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_definition_pins_native_generative_task(tmp_path, monkeypatch, model_type, trust, bypass):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    config = tiny_config()
    payload = config.to_dict()
    payload["model_type"] = model_type
    payload["language_config"] = payload.pop("text_config")
    payload["auto_map"] = {"AutoConfig": "evil.Config", "AutoModel": "evil.Model"}
    marker = tmp_path / "EXECUTED"
    (tmp_path / "evil.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError('checkpoint code')\n")
    (tmp_path / "config.json").write_text(json.dumps(payload))
    selected = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(selected) is UnlimitedOcrConfig
    assert selected.text_config.hidden_size == 32
    with torch.device("meta"):
        model = UnlimitedOCRQModel.loader.from_config(selected, trust_remote_code=trust)
    assert type(model) is UnlimitedOcrForConditionalGeneration
    assert model.can_generate()
    assert UnlimitedOCRQModel.extract_layers_node() == ["model.language_model.layers"]
    assert UnlimitedOCRQModel.require_trust_remote_code is False
    assert not marker.exists()


def test_native_task_forward_image_and_cache():
    torch.manual_seed(731)
    model = UnlimitedOcrForConditionalGeneration(tiny_config()).eval()
    with torch.no_grad():
        model.model.image_newline.normal_(std=0.05)
        model.model.view_separator.normal_(std=0.05)
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    ids = torch.tensor([[4, *([3] * 7), 5]])
    with torch.no_grad():
        first = model(ids, pixel_values=pixels, use_cache=True)
        assert first.image_hidden_states[0].shape == (7, 32)
        continued = model(torch.tensor([[6, 7]]), past_key_values=first.past_key_values,
                          attention_mask=torch.ones(1, 11, dtype=torch.long), use_cache=True)
        full = model(torch.cat((ids, torch.tensor([[6, 7]])), dim=1), pixel_values=pixels, use_cache=False)
    torch.testing.assert_close(continued.logits, full.logits[:, -2:], atol=1e-6, rtol=1e-5)
    assert continued.past_key_values.get_seq_length() == 11


def test_native_legacy_config_does_not_mutate_caller():
    original = tiny_config().to_dict()
    original["language_config"] = original.pop("text_config")
    snapshot = copy.deepcopy(original)
    config = UnlimitedOcrConfig.from_dict(original)
    assert config.text_config.hidden_size == 32
    assert original == snapshot


def test_pinned_assets_integrity():
    manifest = json.loads((FIXTURES.parent / "source_manifest.json").read_bytes())
    for name, digest in manifest["sha256"].items():
        if name.endswith(".py"):
            continue
        data = (FIXTURES.parent / name).read_bytes() if name == "LICENSE" else asset(name)
        assert hashlib.sha256(data).hexdigest() == digest, name


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_actual_public_hostile_load_image_cache_and_two_saves(tmp_path, monkeypatch, legacy, trust, bypass):
    from tokenicer import Tokenicer

    from gptqmodel import BACKEND, GPTQModel

    original_load = Tokenicer.load
    normalizations = []

    def normalize_once(*args, **kwargs):
        normalizations.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(Tokenicer, "load", staticmethod(normalize_once))

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    source = source_fixture(tmp_path / "source", legacy=legacy)
    marker = hostile(tmp_path / "source")
    original_assets = {name: (tmp_path / "source" / name).read_bytes()
                       for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]}
    loaded = GPTQModel.load(
        str(tmp_path / "source"), device="cpu", backend=BACKEND.TORCH,
        dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager",
    )
    from transformers import UnlimitedOcrProcessor
    assert type(loaded.model) is UnlimitedOcrForConditionalGeneration
    assert type(loaded.processor) is UnlimitedOcrProcessor
    from defuser.modeling.update_module import update_module
    update_module(source, cleanup_original=False)
    update_module(loaded.model, cleanup_original=False)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
    row = {"input_ids": torch.tensor([[4, *([3] * 7), 5]]),
           "pixel_values": torch.linspace(-1, 1, 3072).reshape(1, 3, 32, 32)}
    with torch.inference_mode():
        expected = source(**row, use_cache=False).logits
        torch.testing.assert_close(loaded.model(**row, use_cache=False).logits, expected, rtol=0, atol=0)
        tokens = loaded.model.generate(**row, max_new_tokens=2, do_sample=False)
        assert tokens.shape[1] == 11
    for cycle in range(2):
        target = tmp_path / f"save{cycle}"
        loaded.model.save_pretrained(target)
        loaded.tokenizer.save_pretrained(target)
        loaded.processor.save_pretrained(target)
        loaded = GPTQModel.load(str(target), device="cpu", backend=BACKEND.TORCH,
                                dtype=torch.float32, attn_implementation="eager")
        update_module(loaded.model, cleanup_original=False)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(**row, use_cache=False).logits, expected, rtol=0, atol=0)
        assert not list(target.glob("*.bin"))
    assert not marker.exists()
    assert len(normalizations) == 3
    for name, expected_bytes in original_assets.items():
        assert (tmp_path / "source" / name).read_bytes() == expected_bytes


@pytest.mark.parametrize("implementation,kv_heads", [("eager", 4), ("eager", 1), ("sdpa", 4), ("sdpa", 2), ("sdpa", 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_layout_cache_and_safetensors(tmp_path, implementation, kv_heads, dtype):
    config = tiny_config(kv_heads=kv_heads, bias=True, dtype=dtype)
    model = UnlimitedOCRQModel.loader.from_config(config, attn_implementation=implementation).to(dtype=dtype).eval()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    with torch.inference_mode():
        first = model(ids, use_cache=True)
        assert first.past_key_values.get_seq_length() == 6
        continuation = model(torch.tensor([[10, 11]]), past_key_values=first.past_key_values, use_cache=True)
        full = model(torch.cat([ids, torch.tensor([[10, 11]])], dim=1), use_cache=False)
        torch.testing.assert_close(continuation.logits, full.logits[:, -2:], atol=0.004, rtol=0.04)
        expected = model.generate(ids, max_new_tokens=4, num_beams=3, do_sample=False)
    model.save_pretrained(tmp_path)
    restored = UnlimitedOCRQModel.loader.from_pretrained(tmp_path, dtype=dtype, attn_implementation=implementation)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(restored.generate(ids, max_new_tokens=4, num_beams=3, do_sample=False), expected)


def test_gqa_eager_explicit_rejection_and_sdpa_retry(tmp_path):
    model = UnlimitedOcrForConditionalGeneration(tiny_config(kv_heads=2)).eval()
    model.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="repeat grouped KV heads.*sdpa"):
        UnlimitedOCRQModel.loader.from_pretrained(tmp_path, attn_implementation="eager")
    loaded = UnlimitedOCRQModel.loader.from_pretrained(tmp_path, attn_implementation="sdpa")
    with torch.inference_mode():
        assert torch.isfinite(loaded(torch.tensor([[4, 5, 6]])).logits).all()


def independent_decoder(model, ids, positions):
    """Source float64 equations; no native norm/RoPE/attention/router helpers."""
    from defuser.modeling.update_module import update_module
    update_module(model, cleanup_original=False)
    state = {key: value.detach().double() for key, value in model.state_dict().items()}
    config = model.config.text_config
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    width = config.hidden_size // heads

    def linear(x, path):
        return x @ state[path + ".weight"].T + state.get(path + ".bias", 0)

    def norm(x, path):
        return x * (x.square().mean(-1, keepdim=True) + config.rms_norm_eps).rsqrt() * state[path + ".weight"]

    freq = config.rope_parameters["rope_theta"] ** (-torch.arange(0, width, 2, dtype=torch.float64) / width)
    angles = torch.cat([positions[..., None] * freq] * 2, dim=-1).unsqueeze(1)

    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return x * angles.cos() + torch.cat([-b, a], dim=-1) * angles.sin()

    def mlp(x, root):
        gate = linear(x, root + ".gate_proj")
        return linear((gate * gate.sigmoid()) * linear(x, root + ".up_proj"), root + ".down_proj")

    x = state["model.language_model.embed_tokens.weight"][ids]
    for index in range(config.num_hidden_layers):
        root = f"model.language_model.layers.{index}"
        y = norm(x, root + ".input_layernorm")
        q = rotate(linear(y, root + ".self_attn.q_proj").view(*ids.shape, heads, width).transpose(1, 2))
        k = rotate(linear(y, root + ".self_attn.k_proj").view(*ids.shape, kv_heads, width).transpose(1, 2))
        v = linear(y, root + ".self_attn.v_proj").view(*ids.shape, kv_heads, width).transpose(1, 2)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = q @ k.transpose(-1, -2) / width ** 0.5
        mask = torch.arange(ids.shape[1])[None, :] <= torch.arange(ids.shape[1])[:, None]
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        x = x + linear((weights @ v).transpose(1, 2).reshape(*ids.shape, -1), root + ".self_attn.o_proj")
        y = norm(x, root + ".post_attention_layernorm")
        if config.mlp_layer_types[index] == "dense":
            value = mlp(y, root + ".mlp")
        else:
            scores = linear(y, root + ".mlp.gate").softmax(dim=-1)
            if config.topk_method == "group_limited_greedy":
                grouped = scores.view(*ids.shape, config.n_group, -1).amax(dim=-1)
                groups = grouped.topk(config.topk_group, dim=-1).indices
                selected = torch.zeros_like(grouped).scatter(-1, groups, 1).repeat_interleave(config.num_experts // config.n_group, -1)
                scores = scores.masked_fill(selected == 0, 0)
            weights, indices = scores.topk(config.num_experts_per_tok, dim=-1, sorted=False)
            weights = weights * config.routed_scaling_factor
            value = mlp(y, root + ".mlp.shared_experts")
            for expert in range(config.num_experts):
                probability = (weights * (indices == expert)).sum(-1, keepdim=True)
                value = value + probability * mlp(y, root + f".mlp.experts.{expert}")
        x = x + value
    return linear(norm(x, "model.language_model.norm"), "lm_head")


@pytest.mark.parametrize("kv_heads,implementation", [(4, "eager"), (1, "eager"), (2, "sdpa"), (4, "sdpa")])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layout", ["dense", "sparse", "mixed"])
@pytest.mark.parametrize("method", ["greedy", "group_limited_greedy"])
def test_independent_source_decoder_oracle(kv_heads, implementation, bias, layout, method):
    torch.manual_seed(137)
    config = tiny_config(kv_heads, bias, layout=layout)
    config.text_config.topk_method = method
    config.text_config.n_group = 2
    config.text_config.topk_group = 1
    config.text_config.routed_scaling_factor = 1.7
    model = UnlimitedOCRQModel.loader.from_config(config, attn_implementation=implementation).eval()
    ids = torch.tensor([[4, 9, 7, 11, 6, 8], [12, 14, 18, 10, 4, 5]])
    positions = torch.arange(32762, 32768).expand(2, -1)
    expected = independent_decoder(model, ids, positions)
    with torch.inference_mode():
        actual = model(ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("layout", ["dense", "mixed", "sparse"])
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_plain_awq_pack_infer_two_reloads(tmp_path, layout, offload):
    from defuser.modeling.update_module import update_module
    from PIL import Image

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.looper.loop_processor import CalibrationConfig
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = source_fixture(tmp_path / "source", legacy=True, layout=layout, released_vocab=True)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    loaded = GPTQModel.load(str(tmp_path / "source"), quantize_config=qcfg, backend=BACKEND.TORCH,
                            dtype=torch.float16, attn_implementation="eager")
    update_module(source, cleanup_original=False)
    expected_count = {"dense": 14, "mixed": 26, "sparse": 38}[layout]
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    dense = {key: value.to(torch.float16).clone() for key, value in source.state_dict().items()
             if not (key.startswith("model.language_model.layers.") and any(f".{p}." in key for p in projections))}
    image = Image.fromarray(torch.arange(32 * 32 * 3).remainder(256).byte().reshape(32, 32, 3).numpy())
    rows = [loaded.prepare_dataset([{"image": image, "text": "<image>\nFree OCR."}])[0]]
    assert (rows[0]["input_ids"] == 128815).sum() == 7
    assert rows[0]["input_ids"].max() < 129280
    rows[0]["pixel_values"] = rows[0]["pixel_values"].half()
    captured = []
    handle = source.model.language_model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs.get("hidden_states", args[0] if args else None).detach().clone()),
        with_kwargs=True,
    )
    with torch.inference_mode():
        source.half()(**rows[0], use_cache=False)
    handle.remove()
    observed = []
    handle = loaded.model.model.language_model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs.get("hidden_states", args[0] if args else None).detach().clone()),
        with_kwargs=True,
    )
    loaded.quantize(CalibrationConfig(dataset=rows, batch_size=1, min_length=1, sort="desc"), backend=BACKEND.TORCH)
    handle.remove()
    assert observed
    torch.testing.assert_close(observed[0], captured[0], rtol=0, atol=0)
    assert sum(isinstance(module, AwqTorchLinear) for module in loaded.model.modules()) == expected_count
    for key, value in dense.items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    packed = {key: value.detach().clone() for key, value in loaded.model.state_dict().items()}
    with torch.inference_mode():
        expected = loaded.model(**rows[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
    for cycle in range(2):
        target = tmp_path / f"packed{cycle}"
        loaded.save_quantized(str(target))
        from safetensors import safe_open
        extra = {}
        for shard in target.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                if UNUSED_PATCH_NAME in tensors.keys():
                    extra[UNUSED_PATCH_NAME] = tensors.get_tensor(UNUSED_PATCH_NAME)
        torch.testing.assert_close(extra[UNUSED_PATCH_NAME], source_state(source)[UNUSED_PATCH_NAME], rtol=0, atol=0)
        loaded = GPTQModel.load(str(target), device="cpu", backend=BACKEND.TORCH,
                                dtype=torch.float16, attn_implementation="eager")
        assert set(loaded.model.state_dict()) == set(packed)
        for key, value in packed.items():
            torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(**rows[0], use_cache=False).logits, expected, rtol=0, atol=0)
        assert not list(target.glob("*.bin"))


def test_released_tokenizer_and_processor_ids_pixels_two_reload_cycles(tmp_path):
    from PIL import Image

    from gptqmodel import BACKEND, GPTQModel

    source_fixture(tmp_path / "source", released_vocab=True)
    loaded = GPTQModel.load(str(tmp_path / "source"), device="cpu", backend=BACKEND.TORCH,
                            dtype=torch.float32)
    expected_ids = [
        ("Free OCR.", [21431, 126041, 16]),
        ("Hello café 你好!", [19923, 57664, 223, 30594, 3]),
        ("  x\t y\n中文", [223, 1527, 200, 383, 201, 21134]),
        ("<image>\nFree OCR.", [128815, 201, 21431, 126041, 16]),
    ]
    image = Image.new("RGB", (32, 32), (17, 79, 193))
    pixels = (torch.tensor([17, 79, 193], dtype=torch.float32) / 255 - 0.5) / 0.5
    for cycle in range(3):
        for text, expected in expected_ids:
            actual = loaded.tokenizer.encode(text, add_special_tokens=False)
            assert actual == expected
            assert loaded.tokenizer.decode(actual, skip_special_tokens=False) == text
        assert loaded.tokenizer.eos_token_id == 1
        assert loaded.tokenizer.pad_token_id == 2
        generation = loaded.model.generation_config
        assert generation.bos_token_id == 0 and generation.eos_token_id == 1
        assert generation.no_repeat_ngram_size == 35
        assert generation.no_repeat_ngram_window_size == 128
        assert generation.max_new_tokens == 32768 and generation.use_cache
        row = loaded.prepare_dataset([{"image": image}])[0]
        torch.testing.assert_close(row["pixel_values"][0], pixels[:, None, None].expand(3, 32, 32))
        assert (row["input_ids"] == 128815).sum() == 7
        with torch.inference_mode():
            generated = loaded.generate(**row, max_new_tokens=2, use_cache=True, do_sample=False)
            assert generated.shape[-1] == row["input_ids"].shape[-1] + 2
        if cycle < 2:
            target = tmp_path / f"save{cycle}"
            loaded.model.save_pretrained(target)
            loaded.tokenizer.save_pretrained(target)
            loaded.processor.save_pretrained(target)
            loaded = GPTQModel.load(str(target), device="cpu", backend=BACKEND.TORCH, dtype=torch.float32)


def test_public_corrupt_source_repair_and_retry(tmp_path):
    from gptqmodel import BACKEND, GPTQModel

    source = source_fixture(tmp_path, legacy=True)
    path = tmp_path / "model.safetensors"
    original = path.read_bytes()
    path.write_bytes(b"incomplete")
    with pytest.raises(Exception, match="[Hh]eader|safetensor|buffer"):
        GPTQModel.load(str(tmp_path), device="cpu", backend=BACKEND.TORCH, dtype=torch.float32)
    path.write_bytes(original)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", backend=BACKEND.TORCH, dtype=torch.float32,
                            attn_implementation="eager")
    from defuser.modeling.update_module import update_module
    update_module(source, cleanup_original=False)
    update_module(loaded.model, cleanup_original=False)
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], expected, rtol=0, atol=0)


def test_native_eager_gqa_defect_and_owned_rejection():
    config = tiny_config(kv_heads=2)
    model = UnlimitedOcrForConditionalGeneration(config).eval()
    with pytest.raises(RuntimeError, match="size of tensor"):
        model(torch.tensor([[4, 5, 6]]), use_cache=False)
    with pytest.raises(ValueError, match="sdpa"):
        UnlimitedOCRQModel.loader.from_config(config, attn_implementation="eager")


@pytest.mark.parametrize("size,expected_images", [((32, 32), 7), ((64, 32), 17), ((32, 64), 19)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_global_local_image_cache_greedy_beam_and_media_retry(tmp_path, size, expected_images, dtype):
    from PIL import Image

    from gptqmodel import BACKEND, GPTQModel

    source_fixture(tmp_path, released_vocab=True)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", backend=BACKEND.TORCH,
                            dtype=dtype, attn_implementation="sdpa")
    image = Image.fromarray(torch.arange(size[0] * size[1] * 3).remainder(256).byte().reshape(size[1], size[0], 3).numpy())
    for prompt in ["No image marker.", "<image><image>"]:
        with pytest.raises(ValueError, match="exactly one"):
            loaded.prepare_dataset([{"image": image, "text": prompt}])
    row = loaded.prepare_dataset([{"image": image}])[0]
    row = loaded.move_input_capture_example(row, torch.device("cpu"))
    assert (row["input_ids"] == 128815).sum() == expected_images
    with torch.inference_mode():
        prefill = loaded.model(**row, use_cache=True)
        assert prefill.image_hidden_states[0].shape == (expected_images, 32)
        assert torch.isfinite(prefill.logits).all()
        continuation = loaded.model(torch.tensor([[6, 9]]), past_key_values=prefill.past_key_values, use_cache=True).logits
        full_row = dict(row)
        full_row["input_ids"] = torch.cat([row["input_ids"], torch.tensor([[6, 9]])], dim=1)
        full_row["attention_mask"] = torch.ones_like(full_row["input_ids"])
        full = loaded.model(**full_row, use_cache=False).logits
        tolerance = {torch.float32: 3e-6, torch.float16: 0.001, torch.bfloat16: 0.008}[dtype]
        torch.testing.assert_close(continuation, full[:, -2:], atol=tolerance, rtol=2e-5)
        for beams in [1, 3]:
            native = loaded.generate(**row, max_new_tokens=3, do_sample=False, use_cache=True, num_beams=beams,
                                     return_dict_in_generate=True, output_scores=True)
            replay = loaded.generate(**row, max_new_tokens=3, do_sample=False, use_cache=False, num_beams=beams,
                                     return_dict_in_generate=True, output_scores=True)
            torch.testing.assert_close(native.sequences, replay.sequences)
            for actual, expected in zip(native.scores, replay.scores):
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=1e-5)
