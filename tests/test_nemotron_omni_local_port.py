# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Released Nemotron Omni metadata cannot select checkpoint Python."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import NemotronHForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.definitions.nemotron_omni import NemotronOmniQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.nemotron_omni.configuration import NemotronOmniConfig
from gptqmodel.models.ports.nemotron_omni.modeling import NemotronH_Omni_Reasoning_V3
from gptqmodel.models.ports.nemotron_omni.processing import NemotronOmniProcessor
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config


PORT = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/nemotron_omni"


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(kv=2, bias=False, audio=True, **kwargs):
    text = {
        "model_type": "nemotron_h", "vocab_size": 64, "hidden_size": 32,
        "layers_block_type": ["mamba", "attention", "mlp", "moe"], "num_attention_heads": 4,
        "num_key_value_heads": kv, "head_dim": 8, "intermediate_size": 64,
        "mamba_num_heads": 8, "mamba_head_dim": 8, "ssm_state_size": 4, "n_groups": 2,
        "conv_kernel": 3, "chunk_size": 4, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32, "moe_shared_expert_intermediate_size": 64,
        "num_logits_to_keep": 0, "attention_bias": bias, "mlp_bias": bias,
    }
    vision = {
        "model_type": "radio", "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 4,
        "max_img_size": 32, "patch_size": 2, "num_cls_tokens": 2, "num_registers": 1,
        "video_temporal_patch_size": 2,
    }
    sound = {
        "model_type": "parakeet_encoder", "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 4, "num_hidden_layers": 1, "num_mel_bins": 16,
        "conv_kernel_size": 3, "subsampling_conv_channels": 8, "subsampling_factor": 8,
        "projection_hidden_size": 32, "projection_bias": False,
    } if audio else None
    config = NemotronOmniConfig(text_config=text, vision_config=vision, audio_config=sound,
                               vision_hidden_size=16, projector_hidden_size=64,
                               image_token_id=18, video_token_id=19, audio_token_id=20, **kwargs)
    config._attn_implementation = "eager"
    return config


def tiny_model(**kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4123)
        return NemotronH_Omni_Reasoning_V3(tiny_config(**kwargs)).eval()


def checkpoint(path, **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    model = tiny_model(**kwargs)
    model.save_pretrained(path)
    vocab = {f"t{i}": i for i in range(64)}
    for index, name in ((18, "<image>"), (19, "<video>"), (20, "<so_embedding>"),
                        (21, "<img>"), (22, "</img>"), (23, "<so_start>"), (24, "<so_end>")):
        del vocab[f"t{index}"]
        vocab[name] = index
    raw = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="t0", pad_token="t0", bos_token="t1",
                                       eos_token="t2", additional_special_tokens=list(vocab)[-7:])
    tokenizer.chat_template = (
        "{% for m in messages %}{% for item in m.content %}"
        "{% if item.type == 'image' %}<image>{% else %}{{ item.text }}{% endif %}"
        "{% endfor %}{% endfor %}"
    )
    tokenizer.save_pretrained(path)
    (path / "preprocessor_config.json").write_text(json.dumps({
        "patch_size": 2, "min_num_patches": 16, "max_num_patches": 16,
        "image_processor_type": "HostileProcessor", "norm_mean": [0.4, 0.5, 0.6], "norm_std": [0.2, 0.3, 0.4],
        "auto_map": {"AutoProcessor": "hostile.Forbidden", "AutoImageProcessor": "hostile.Forbidden"},
    }))
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n"
    )
    for filename, mappings in (
        ("config.json", ("AutoConfig", "AutoModel", "AutoModelForCausalLM")),
        ("tokenizer_config.json", ("AutoTokenizer",)),
    ):
        content = json.loads((path / filename).read_text())
        content["auto_map"] = dict.fromkeys(mappings, "hostile.Forbidden")
        (path / filename).write_text(json.dumps(content))
    return model, marker


def image_inputs():
    return {"input_ids": torch.tensor([[1, 18, 18, 18, 18, 3]]),
            "pixel_values": torch.linspace(-1.3, 1.4, 16 * 12).reshape(16, 12),
            "image_grid_hw": torch.tensor([[4, 4]])}


@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_native_parameter_owners_and_text_parity(kv, bias):
    model = tiny_model(kv=kv, bias=bias)
    assert type(model.language_model) is NemotronHForCausalLM
    reference = NemotronHForCausalLM(model.config.text_config).eval()
    reference.load_state_dict(model.language_model.state_dict(), strict=True)
    ids = torch.tensor([[1, 4, 7, 9, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, reference(ids).logits, atol=0, rtol=0)
    assert model(ids, return_dict=False)[0].shape == (1, 5, 64)


@pytest.mark.parametrize("ps_version", ["v1", "v2"])
def test_pixel_shuffle_matches_source_equations_and_independent_projector(ps_version):
    model = tiny_model(ps_version=ps_version)
    # Rectangular patches expose v1/v2's spatial transpose (square inputs cannot).
    features = torch.arange(2 * 4 * 6 * 16, dtype=torch.float64).reshape(2, 4, 6, 16) / 100
    expected = features.reshape(2, 4, 3, 32).permute(0, 2, 1, 3).contiguous().reshape(2, 3, 2, 64)
    if ps_version == "v2":
        expected = expected.permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(model.pixel_shuffle(features), expected, atol=0, rtol=0)
    weight = model.multi_modal_projector.layer_norm.weight.double()
    normed = (expected.float() * (expected.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()).double() * weight
    hidden = F.linear(normed, model.multi_modal_projector.linear_1.weight.double()).relu().square()
    oracle = F.linear(hidden, model.multi_modal_projector.linear_2.weight.double()).flatten(1, 2)
    model.double()
    actual = model.project_vision_features(features.flatten(1, 2), 4, 6)
    torch.testing.assert_close(actual, oracle, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_image_cache_generation_and_safetensors_roundtrip(tmp_path, dtype, implementation):
    source = tmp_path / "source"
    tiny_model().save_pretrained(source)
    model = NemotronH_Omni_Reasoning_V3.from_pretrained(source, dtype=dtype,
                                                     attn_implementation=implementation).eval()
    inputs = image_inputs()
    with torch.inference_mode():
        full = model(**inputs)
        assert torch.isfinite(full.logits).all()
        continuation = model(torch.tensor([[7]]), past_key_values=full.past_key_values,
                             attention_mask=torch.ones(1, 7, dtype=torch.long))
        joined = {**inputs, "input_ids": torch.cat((inputs["input_ids"], torch.tensor([[7]])), dim=1)}
        direct = model(**joined, use_cache=False)
        torch.testing.assert_close(continuation.logits[:, -1], direct.logits[:, -1], atol=0.015, rtol=0.015)
        greedy = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        beam = model.generate(**inputs, max_new_tokens=3, do_sample=False, num_beams=2)
    model.save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
    reloaded = NemotronH_Omni_Reasoning_V3.from_pretrained(tmp_path, dtype=dtype, attn_implementation=implementation).eval()
    assert set(reloaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, atol=0, rtol=0)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(**inputs).logits, full.logits, atol=0, rtol=0)
        assert torch.equal(reloaded.generate(**inputs, max_new_tokens=3, do_sample=False), greedy)
        assert torch.equal(reloaded.generate(**inputs, max_new_tokens=3, do_sample=False, num_beams=2), beam)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("legacy", [False, True])
def test_public_model_tokenizer_processor_ignores_hostile_maps(tmp_path, monkeypatch, trust, bypass, legacy):
    expected, marker = legacy_checkpoint(tmp_path)[:2] if legacy else checkpoint(tmp_path)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    actual = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                           attn_implementation="eager")
    assert type(actual) is NemotronOmniQModel
    assert type(actual.model) is NemotronH_Omni_Reasoning_V3
    assert type(actual.processor) is NemotronOmniProcessor
    assert actual.processor.tokenizer is actual.tokenizer.tokenizer
    with torch.inference_mode():
        torch.testing.assert_close(actual.model(**image_inputs()).logits, expected(**image_inputs()).logits,
                                   rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("asset", ["tokenizer.json", "preprocessor_config.json"])
def test_public_malformed_media_asset_is_fail_closed_then_repairs_on_retry(tmp_path, monkeypatch, asset):
    _, marker = checkpoint(tmp_path)
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    filename = tmp_path / asset
    original = filename.read_bytes()
    filename.write_text("{broken JSON")
    with pytest.raises(Exception, match="JSON|line"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    assert not marker.exists()
    filename.write_bytes(original)
    repaired = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    assert type(repaired.processor) is NemotronOmniProcessor
    with torch.inference_mode():
        assert torch.isfinite(repaired.model(**image_inputs()).logits).all()
    assert not marker.exists()


@pytest.mark.parametrize("pruning", [0.0, 0.7])
@pytest.mark.parametrize("frames", [1, 3, 4])
def test_video_temporal_padding_pruning_source_oracle_and_generation(pruning, frames):
    model = tiny_model(video_pruning_rate=pruning)
    video = torch.linspace(-1.3, 1.5, frames * 3 * 8 * 12).reshape(frames, 3, 8, 12)
    padded = torch.cat((video, video[-1:])) if frames % 2 else video
    # Source packs frame/channel row-major, not channel/frame interleaved.
    packed = padded.reshape(-1, 6, 8, 12)
    with torch.inference_mode():
        vision = model.vision_model(packed).features
        x = vision.reshape(-1, 4, 6, 16).reshape(-1, 4, 3, 32).transpose(1, 2).contiguous()
        x = x.reshape(-1, 3, 2, 64).transpose(1, 2).contiguous().flatten(1, 2)
        expected = model.multi_modal_projector(x)
        torch.testing.assert_close(model.get_video_features(video).pooler_output, expected, atol=0, rtol=0)
        count = expected.shape[0] * 6
        ids = torch.tensor([[1] + [18] * count + [3]])
        keep = torch.ones(count, dtype=torch.bool)
        if pruning:
            scores = torch.ones(expected.shape[:2]) * 255
            scores[1:] = 1 - F.cosine_similarity(expected[1:], expected[:-1], dim=-1)
            keep.zero_()
            keep[torch.argsort(scores.flatten(), descending=True, stable=True)[:max(6, int(count * (1 - pruning)))]] = True
        retained = torch.ones(ids.shape[1], dtype=torch.bool)
        retained[1:-1] = keep
        selected = ids[:, retained]
        embeds = model.get_input_embeddings()(selected)
        embeds[:, 1:-1] = expected.flatten(0, 1)[keep]
        oracle = model.language_model(inputs_embeds=embeds, use_cache=False).logits
        actual = model(input_ids=ids, pixel_values_videos=video, attention_mask=torch.ones_like(ids),
                       labels=ids, use_cache=False)
        torch.testing.assert_close(actual.logits, oracle, atol=0, rtol=0)
        assert torch.isfinite(actual.loss)
        generated = model.generate(input_ids=ids, pixel_values_videos=video, max_new_tokens=3, do_sample=False)
        manual = selected.clone()
        for _ in range(3):
            token = model.language_model(inputs_embeds=embeds, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            manual = torch.cat((manual, token), dim=1)
            embeds = torch.cat((embeds, model.get_input_embeddings()(token)), dim=1)
        assert torch.equal(generated, manual)
        assert model.generate(input_ids=ids, pixel_values_videos=video, max_new_tokens=3, num_beams=2).shape == generated.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_audio_ragged_clips_and_mixed_image_forward_cache_generation(tmp_path, dtype):
    source = tmp_path / "source"
    tiny_model().save_pretrained(source)
    model = NemotronH_Omni_Reasoning_V3.from_pretrained(source, dtype=dtype).eval()
    features = torch.linspace(-0.9, 1.1, 2 * 33 * 16).reshape(2, 33, 16).to(dtype)
    mask = torch.ones(2, 33, dtype=torch.long)
    mask[1, 18:] = 0
    with torch.inference_mode():
        native = model.audio_tower(features, attention_mask=mask).last_hidden_state
        lengths = model.audio_tower._get_subsampling_output_length(mask.sum(-1) + 1)
        projected = model.embed_audio(native)
        oracle = torch.cat([projected[i, :length] for i, length in enumerate(lengths)])
        torch.testing.assert_close(model.get_audio_features(features, mask).pooler_output, oracle, atol=0, rtol=0)
        image = image_inputs()
        ids = torch.cat((image["input_ids"], torch.full((1, oracle.shape[0]), 20)), dim=1)
        inputs = {**image, "input_ids": ids, "input_features": features, "input_features_mask": mask}
        full = model(**inputs)
        assert torch.isfinite(full.logits).all()
        continuation = model(torch.tensor([[7]]), past_key_values=full.past_key_values)
        joined = model(**{**inputs, "input_ids": torch.cat((ids, torch.tensor([[7]])), dim=1)}, use_cache=False)
        torch.testing.assert_close(continuation.logits[:, -1], joined.logits[:, -1], atol=.015, rtol=.015)
        assert model.generate(**inputs, max_new_tokens=3, eos_token_id=None).shape[1] == ids.shape[1] + 3


def legacy_state(model):
    """Independent source-format writer for exercising real legacy input."""
    import re

    state = {}
    for name, value in model.state_dict().items():
        name = name.replace("language_model.model.", "language_model.backbone.")
        for old, new in (("multi_modal_projector.layer_norm.", "mlp1.0."),
                         ("multi_modal_projector.linear_1.", "mlp1.1."),
                         ("multi_modal_projector.linear_2.", "mlp1.3."),
                         ("audio_tower.", "sound_encoder.encoder."),
                         ("embed_audio.layer_norm.", "sound_projection.norm."),
                         ("embed_audio.linear_1.", "sound_projection.linear1."),
                         ("embed_audio.linear_2.", "sound_projection.linear2."),
                         ("vision_model.embeddings.video_patch_projection", "vision_model.radio_model.model.patch_generator.video_embedder"),
                         ("vision_model.embeddings.patch_projection", "vision_model.radio_model.model.patch_generator.embedder"),
                         ("vision_model.embeddings.position_embedding", "vision_model.radio_model.model.patch_generator.pos_embed"),
                         ("vision_model.embeddings.cls_register_token", "vision_model.radio_model.model.patch_generator.cls_token.token"),
                         ("vision_model.encoder.layer.", "vision_model.radio_model.model.blocks."),
                         (".attention.o_proj.", ".attn.proj.")):
            name = name.replace(old, new)
        if name != "vision_model.summary_idxs":
            state[name] = value.contiguous()
    for name in list(state):
        if ".attention.q_proj." in name:
            prefix, suffix = name.split(".attention.q_proj.")
            state[prefix + ".attn.qkv." + suffix] = torch.cat([
                state.pop(prefix + f".attention.{p}_proj." + suffix) for p in "qkv"
            ]).contiguous()
    assert not any(re.search(r"\.attention\.[qkv]_proj\.", k) for k in state if k.startswith("vision_model"))
    return state


def legacy_checkpoint(path):
    from safetensors.torch import save_file

    model, marker = checkpoint(path, legacy_radio_layout=True)
    _convert_model_with_defuser(NemotronOmniQModel, model, cleanup_original=False)
    state = legacy_state(model)
    for name, shape in (("vision_model.radio_model.input_conditioner.norm_mean", (3, 1, 1)),
                        ("vision_model.radio_model.input_conditioner.norm_std", (3, 1, 1)),
                        ("sound_encoder.encoder.feature_extractor.featurizer.fb", (1, 16, 257)),
                        ("sound_encoder.encoder.feature_extractor.featurizer.window", (400,))):
        state[name] = torch.linspace(.01, .99, torch.Size(shape).numel()).reshape(shape)
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    config = json.loads((path / "config.json").read_text())
    config["llm_config"] = config.pop("text_config")
    config["sound_config"] = config.pop("audio_config")
    config["sound_config"]["model_type"] = "parakeet"
    (path / "config.json").write_text(json.dumps(config))
    return model, marker, state


def test_source_format_load_conversion_and_corrupt_tensor_repair_retry(tmp_path):
    expected, marker, state = legacy_checkpoint(tmp_path)
    actual = GPTQModel.load(str(tmp_path), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        torch.testing.assert_close(actual.model(**image_inputs()).logits, expected(**image_inputs()).logits,
                                   atol=1e-6, rtol=1e-6)
    from safetensors import SafetensorError

    tensor_path = tmp_path / "model.safetensors"
    saved = tensor_path.read_bytes()
    tensor_path.write_bytes(b"broken safetensors header")
    with pytest.raises((SafetensorError, OSError), match="header|safetensors"):
        GPTQModel.load(str(tmp_path), dtype=torch.float32, device="cpu")
    tensor_path.write_bytes(saved)
    repaired = GPTQModel.load(str(tmp_path), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        assert torch.isfinite(repaired.model(**image_inputs()).logits).all()
    assert not marker.exists()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("offload", [False, True])
def test_public_plain_awq_image_calibration_all_owners_and_two_reloads(tmp_path, legacy, offload):
    source = tmp_path / "source"
    expected, marker = legacy_checkpoint(source)[:2] if legacy else checkpoint(source)
    cfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload,
                    offload_to_disk_path=str(tmp_path / "offload"))
    model = GPTQModel.load(str(source), quantize_config=cfg, dtype=torch.float16, backend=BACKEND.TORCH)
    baseline = GPTQModel.load(str(source), dtype=torch.float16, device="cpu")
    _convert_model_with_defuser(NemotronOmniQModel, baseline.model, cleanup_original=False)
    dense = {k: v.clone() for k, v in baseline.model.state_dict().items()
             if not k.startswith("language_model.model.layers.") or ".gate." in k or ".norm" in k or ".conv1d." in k
             or k.endswith(("A_log", ".D", "dt_bias"))}
    modules = {name for name, module in model.model.named_modules() if isinstance(module, torch.nn.Linear)
               and name.startswith("language_model.model.layers.") and not name.endswith(".gate")}
    assert len(modules) == 18
    from PIL import Image

    calibration = [[{"role": "user", "content": [
        {"type": "image", "image": Image.new("RGB", (8, 8), color=(110, 40, 190))},
        {"type": "text", "text": "t3 t4 t5"},
    ]}] for _ in range(2)]
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == modules
    assert all(not hasattr(model.model.get_submodule(name), "weight") for name in packed)
    for key, value in dense.items():
        torch.testing.assert_close(model.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        logits = model.model(**image_inputs(), use_cache=False).logits
        assert torch.isfinite(logits).all()
    for cycle in range(2):
        target = tmp_path / f"quantized-{cycle}"
        model.save(str(target))
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        reloaded = GPTQModel.load(str(target), dtype=torch.float16, backend=BACKEND.TORCH, device="cpu")
        assert type(reloaded.model) is NemotronH_Omni_Reasoning_V3
        assert type(reloaded.processor) is NemotronOmniProcessor
        assert set(reloaded.model.state_dict()) == set(model.model.state_dict())
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(reloaded.model.state_dict()[key], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded.model(**image_inputs(), use_cache=False).logits, logits, atol=0, rtol=0)
        model = reloaded
    assert not marker.exists()


def test_released_legacy_config_uses_owned_native_equivalent(tmp_path):
    payload = json.loads((PORT / "source_config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(payload))
    config = load_model_config(tmp_path, trust_remote_code=False)
    assert type(config).__module__.startswith("gptqmodel.models.ports.nemotron_omni")
    assert config.text_config.model_type == "nemotron_h"
    assert config.vision_config.model_type == "radio"
    assert config.audio_config.model_type == "parakeet_encoder"
    assert config.vision_config.num_cls_tokens == 3
    assert config.vision_config.num_registers == 7
    assert config.audio_config.attention_bias is False
    assert config.audio_config.scale_input is False
    assert NemotronOmniQModel.require_trust_remote_code is False


def test_outer_config_return_dict_false_preserves_native_tuple_contract():
    model = tiny_model()
    model.config.return_dict = False
    with torch.inference_mode():
        output = model(**image_inputs(), use_cache=False)
        named = model(**image_inputs(), use_cache=False, return_dict=True)
    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], named.logits, atol=0, rtol=0)
