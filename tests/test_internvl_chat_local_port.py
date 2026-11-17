# SPDX-License-Identifier: Apache-2.0
"""Real legacy wrapper qualification; no checkpoint Python is imported."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import DynamicCache, StaticCache

from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.internvl_chat.configuration_internvl_chat import InternVLChatConfig
from gptqmodel.models.ports.internvl_chat.modeling_internvl_chat import InternVLChatModel, decoder_model_class


ARCHITECTURES = ["LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Phi3ForCausalLM", "InternLM2ForCausalLM"]


def tiny_config(architecture="Qwen3ForCausalLM", **kwargs):
    llm = {"architectures": [architecture], "vocab_size": 48, "hidden_size": 16, "intermediate_size": 24,
               "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
               "head_dim": 8, "max_position_embeddings": 64, "original_max_position_embeddings": 32,
               "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
               "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 24}
    vision = {"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 2,
                  "image_size": 8, "patch_size": 2, "use_flash_attn": False, "norm_type": "layer_norm", "qk_normalization": False}
    vision.update(kwargs.pop("vision", {}))
    llm.update(kwargs.pop("llm", {}))
    return InternVLChatConfig(vision_config=vision, llm_config=llm, ps_version=kwargs.pop("ps_version", "v2"), **kwargs)


def save_tiny_checkpoint(path, architecture="Qwen3ForCausalLM", **kwargs):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model = InternVLChatModel(tiny_config(architecture, **kwargs)).eval()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "[UNK]": 4}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<img>", "</img>", "<IMG_CONTEXT>", "<|im_end|>", "<|im_start|>"]})
    tokenizer.save_pretrained(path)
    return model


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_gptq_load_uses_local_config_and_model_with_hostile_automap(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    original = save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "malicious.Config", "AutoModel": "malicious.Model", "AutoModelForCausalLM": "malicious.Model"}
    config["llm_config"]["auto_map"] = config["auto_map"]
    config_path.write_text(json.dumps(config))
    (tmp_path / "malicious.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is InternVLChatQModel
    assert type(loaded.model) is InternVLChatModel and type(loaded.model.config) is InternVLChatConfig
    ids = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(input_ids=ids).logits, original(input_ids=ids).logits, atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_public_plain_awq_packs_and_reloads_actual_decoder_weights(tmp_path, architecture):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    original = save_tiny_checkpoint(tmp_path, architecture, llm={"hidden_size": 32, "intermediate_size": 64,
                                    "head_dim": 16, "moe_intermediate_size": 64})
    vision_before = {name: value.clone() for name, value in original.vision_model.state_dict().items()}
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None,
                            offload_to_disk=False), backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    expected_count = {"InternLM2ForCausalLM": 10, "Phi3ForCausalLM": 10, "Qwen3MoeForCausalLM": 20}.get(architecture, 14)
    packed = {name: module for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == expected_count
    assert all(name.startswith("language_model.model.layers.") for name in packed)
    assert all(module.qweight.dtype == torch.int32 for module in packed.values())
    for name, tensor in wrapper.model.vision_model.state_dict().items():
        torch.testing.assert_close(tensor, vision_before[name].to(tensor.dtype), atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert type(reloaded.model) is InternVLChatModel
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == expected_count
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_pil_conversation_preparation_and_image_capture(tmp_path, architecture):
    from PIL import Image

    from gptqmodel import GPTQModel

    save_tiny_checkpoint(tmp_path, architecture, max_dynamic_patch=2, use_thumbnail=True)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    conversation = [{"role": "user", "content": [Image.new("RGB", (16, 8), color=(35, 71, 143)),
                                                 {"type": "text", "text": "What is in this image?"}]}]
    batch = wrapper.prepare_inputs_for_conversation(conversation)
    assert batch["pixel_values"].shape == (3, 3, 8, 8)  # two tiles and the thumbnail
    assert torch.isfinite(batch["pixel_values"]).all()
    assert int((batch["input_ids"] == batch["img_context_token_id"]).sum()) == 12
    moved = wrapper.move_input_capture_example(batch, torch.device("cpu"))
    with torch.inference_mode():
        actual = wrapper.run_input_capture(moved, use_cache=False, data_device=torch.device("cpu"))
        expected = wrapper.model.generate(pixel_values=moved["pixel_values"], input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"], max_new_tokens=1, do_sample=False,
                    eos_token_id=batch["eos_token_id"], pad_token_id=0)
    assert actual.shape == (1, 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_decoder_multimodal_oracle_cache_generation_and_reload(architecture, tmp_path):
    torch.manual_seed(19)
    model = InternVLChatModel(tiny_config(architecture)).eval()
    model.img_context_token_id = 3
    ids = torch.tensor([[7, 3, 3, 3, 3, 8]])
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        # Native decoder is the numerical oracle; insertion is independently
        # performed at the checkpoint-defined context positions.
        expected_embeds = model.language_model.get_input_embeddings()(ids).clone()
        expected_embeds[:, 1:5] = model.extract_feature(pixels)
        oracle = model.language_model(inputs_embeds=expected_embeds, use_cache=False).logits
        result = model(pixel_values=pixels, input_ids=ids, use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(result.logits, oracle, atol=1e-6, rtol=1e-5)
        assert type(model.language_model) is decoder_model_class(architecture)
        assert result.hidden_states is not None
        continued = model(input_ids=torch.tensor([[9]]), past_key_values=result.past_key_values,
                          attention_mask=torch.ones(1, 7), use_cache=True).logits
        full = model(pixel_values=pixels, input_ids=torch.cat((ids, torch.tensor([[9]])), dim=1)).logits
        torch.testing.assert_close(continued[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)
        assert result.past_key_values.get_seq_length() == 7
        generated = model.generate(pixel_values=pixels, input_ids=ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
        direct = model.language_model.generate(inputs_embeds=expected_embeds, max_new_tokens=2, do_sample=False, eos_token_id=None)
        assert torch.equal(generated, direct)
        assert generated.shape == (1, 2)
        for cache_class in (DynamicCache, StaticCache):
            cache = cache_class(config=model.config.llm_config, **({"max_cache_len": 12} if cache_class is StaticCache else {}))
            model(pixel_values=pixels, input_ids=ids, past_key_values=cache, use_cache=True)
            continuation = model(input_ids=torch.tensor([[9]]), attention_mask=torch.ones(1, 7),
                                 past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(continuation[:, -1], full[:, -1], atol=1e-6, rtol=1e-5)
            cache.reset()
            short_ids = torch.tensor([[7, 8]])
            retried = model(input_ids=short_ids, past_key_values=cache, use_cache=True).logits
            torch.testing.assert_close(retried, model(input_ids=short_ids, use_cache=False).logits, atol=1e-6, rtol=1e-5)
        beam = model.generate(pixel_values=pixels, input_ids=ids, max_new_tokens=2, num_beams=2, do_sample=False, eos_token_id=None)
        direct_beam = model.language_model.generate(inputs_embeds=expected_embeds, max_new_tokens=2, num_beams=2,
                                                   do_sample=False, eos_token_id=None)
        assert torch.equal(beam, direct_beam)
        assert model(input_ids=torch.tensor([[7, 8]]), labels=torch.tensor([[7, 8]])).loss.isfinite()
        assert isinstance(model(input_ids=ids, return_dict=False), tuple)
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert not list(tmp_path.glob("*.bin"))
    saved = load_file(tmp_path / "model.safetensors")
    if architecture != "Qwen3MoeForCausalLM":
        assert set(saved) == set(model.state_dict())
    else:
        # TF5 packs experts at runtime and saves the original checkpoint's
        # per-expert projections; those checkpoint owners remain the contract.
        for i in range(2):
            for expert in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    assert f"language_model.model.layers.{i}.mlp.experts.{expert}.{proj}.weight" in saved
    restored, info = InternVLChatQModel.loader.from_pretrained(tmp_path, output_loading_info=True, dtype=torch.float32)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    restored.eval().img_context_token_id = 3
    with torch.no_grad():
        torch.testing.assert_close(restored(pixel_values=pixels, input_ids=ids).logits, oracle, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_tied_embedding_checkpoint_aliases_reload_without_renaming(architecture, tmp_path):
    model = InternVLChatModel(tiny_config(architecture, llm={"tie_word_embeddings": True})).eval()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    ids = torch.tensor([[7, 8, 9]])
    expected = model(input_ids=ids).logits
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored, info = InternVLChatQModel.loader.from_pretrained(tmp_path, output_loading_info=True, dtype=torch.float32)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    torch.testing.assert_close(restored(input_ids=ids).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_local_selection_ignores_malicious_automap_and_trust(tmp_path, monkeypatch, trust, bypass):
    # Existing security bypass is explicitly part of the public loader contract.
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    cfg = tiny_config().to_dict()
    cfg["auto_map"] = {"AutoConfig": "malicious.Config", "AutoModel": "malicious.Model"}
    cfg["llm_config"]["auto_map"] = cfg["auto_map"]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "malicious.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    loader = InternVLChatQModel.loader
    assert isinstance(loader, LocalModelLoader)
    config = loader.load_config(tmp_path, trust_remote_code=trust, code_revision="bad")
    assert type(config) is InternVLChatConfig
    with torch.device("meta"):
        model = loader.from_config(config, trust_remote_code=trust, code_revision="bad")
    assert type(model) is InternVLChatModel
    assert not InternVLChatQModel.require_trust_remote_code


def test_image_failure_then_retry_preserves_state_and_matches_clean_model():
    model = InternVLChatModel(tiny_config()).eval()
    model.img_context_token_id = 3
    pixels = torch.randn(1, 3, 8, 8)
    ids = torch.tensor([[3, 3, 3, 3, 8]])
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="token count"):
        model(pixel_values=pixels, input_ids=ids[:, 1:])
    with torch.no_grad():
        good = model(pixel_values=pixels, input_ids=ids).logits
        flagged = model(pixel_values=torch.cat((pixels, pixels * 2)), input_ids=ids,
                        image_flags=torch.tensor([[1], [0]])).logits
    torch.testing.assert_close(good, flagged, atol=1e-6, rtol=1e-5)
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("changes,match", [
    ({"llm": {"architectures": ["MaliciousModel"]}}, "Unsupported"),
    ({"llm": {"model_type": "llama"}}, "mismatch"),
    ({"downsample_ratio": 0.3}, "reciprocal"),
    ({"force_image_size": 7}, "divisible"),
    ({"ps_version": "bad"}, "ps_version"),
])
def test_invalid_config_rejected_before_model_allocation(changes, match):
    with pytest.raises(ValueError, match=match):
        tiny_config(**changes)


def test_mutated_config_revalidated_before_allocation():
    cfg = tiny_config()
    cfg.llm_config.architectures = ["Unknown"]
    with pytest.raises(ValueError, match="Unsupported"):
        InternVLChatModel(cfg)


def test_pinned_1b_checkpoint_parameter_names_and_shapes_on_meta():
    import gptqmodel.models.ports.internvl_chat as port

    root = Path(port.__file__).parent
    cfg = InternVLChatConfig(**json.loads((root / "checkpoint_config.json").read_text()))
    with torch.device("meta"):
        model = InternVLChatModel(cfg)
    manifest = json.loads((root / "checkpoint_manifest.json").read_text())
    assert set(model.state_dict()) == set(manifest)
    for key, value in model.state_dict().items():
        assert list(value.shape) == manifest[key]["shape"], key


def _vision_oracle(model, pixels):
    """Independent functional eager equations from the pinned vision source."""
    state = model.state_dict()
    cfg = model.config
    patch = F.conv2d(pixels, state["embeddings.patch_embedding.weight"],
                     state["embeddings.patch_embedding.bias"], stride=cfg.patch_size)
    grid = patch.shape[-2:]
    x = patch.flatten(2).transpose(1, 2)
    x = torch.cat((state["embeddings.class_embedding"].expand(x.shape[0], -1, -1), x), dim=1)
    pos = state["embeddings.position_embedding"]
    side = cfg.image_size // cfg.patch_size
    pos_patch = pos[:, 1:].float().reshape(1, side, side, -1).permute(0, 3, 1, 2)
    pos_patch = F.interpolate(pos_patch, size=grid, mode="bicubic", align_corners=False)
    pos_patch = pos_patch.flatten(2).transpose(1, 2).to(pos.dtype)
    x = x + torch.cat((pos[:, :1], pos_patch), dim=1)
    states = [x]

    def norm(x, name):
        weight = state[name + ".weight"]
        if name + ".bias" in state:
            return F.layer_norm(x, (x.shape[-1],), weight, state[name + ".bias"], cfg.layer_norm_eps)
        y = x.float()
        return weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + cfg.layer_norm_eps)).to(x.dtype)

    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state.get(name + ".bias"))

    for index in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{index}"
        y = norm(x, p + ".norm1")
        b, n, c = y.shape
        q, k, v = linear(y, p + ".attn.qkv").reshape(b, n, 3, cfg.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        if cfg.qk_normalization:
            q = norm(q.transpose(1, 2).flatten(-2), p + ".attn.q_norm").reshape(b, n, cfg.num_attention_heads, -1).transpose(1, 2)
            k = norm(k.transpose(1, 2).flatten(-2), p + ".attn.k_norm").reshape(b, n, cfg.num_attention_heads, -1).transpose(1, 2)
        score = ((q * (c // cfg.num_attention_heads) ** -0.5) @ k.transpose(-1, -2)).softmax(-1)
        y = (score @ v).transpose(1, 2).reshape(b, n, c)
        x = x + linear(y, p + ".attn.proj") * state[p + ".ls1"]
        y = F.gelu(linear(norm(x, p + ".norm2"), p + ".mlp.fc1"))
        x = x + linear(y, p + ".mlp.fc2") * state[p + ".ls2"]
        states.append(x)
    return states


@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("qk", [False, True])
@pytest.mark.parametrize("ps_version", ["v1", "v2"])
def test_vision_computation_grid_order_and_hidden_layer_selection(norm, qk, ps_version):
    model = InternVLChatModel(tiny_config(vision={"norm_type": norm, "qk_normalization": qk},
                                         ps_version=ps_version, select_layer=1)).eval()
    # Different image size exercises learned positional-grid interpolation.
    pixels = torch.randn(2, 3, 12, 12)
    with torch.no_grad():
        oracle_states = _vision_oracle(model.vision_model, pixels)
        output = model.vision_model(pixels, output_hidden_states=True)
        for actual, expected in zip(output.hidden_states, oracle_states):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        # Source pixel shuffle is independently expressed by spatial grouping.
        grid = oracle_states[1][:, 1:].reshape(2, 6, 6, 16)
        oracle = grid.reshape(2, 3, 2, 3, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(2, 3, 3, 64)
        if ps_version == "v1":
            oracle = oracle.transpose(1, 2)
        oracle = model.mlp1(oracle.reshape(2, 9, 64))
        torch.testing.assert_close(model.extract_feature(pixels), oracle, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["su", "yarn"])
def test_legacy_phi3_rope_short_long_and_reset_matches_equations(kind):
    import math

    cfg = tiny_config("Phi3ForCausalLM", llm={"rope_scaling": {
        "type": kind, "short_factor": [1.0] * 4, "long_factor": [2.0] * 4}})
    model = InternVLChatModel(cfg).eval()
    rope = model.language_model.model.rotary_emb
    x = torch.zeros(1, 1, 16)
    scale = 64 / 32
    amplitude = 0.1 * math.log(scale) + 1 if kind == "yarn" else math.sqrt(1 + math.log(scale) / math.log(32))
    for length in (4, 40, 4):
        pos = torch.arange(length).unsqueeze(0)
        cos, sin = rope(x, pos)
        inv = 1 / ((1 if length <= 32 else 2) * 10000 ** (torch.arange(0, 8, 2).float() / 8))
        freq = pos.unsqueeze(-1) * inv
        theta = torch.cat((freq, freq), dim=-1)
        torch.testing.assert_close(cos, theta.cos() * amplitude, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sin, theta.sin() * amplitude, atol=1e-6, rtol=1e-5)
