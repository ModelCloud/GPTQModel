# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Offline qualification of native Apertus and the definition-owned 1.5 port."""

import copy
import json

import pytest
import torch
from huggingface_hub.errors import StrictDataclassClassValidationError
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import ApertusConfig, ApertusForCausalLM, PreTrainedTokenizerFast

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.apertus.configuration_apertus1p5 import Apertus1p5Config, Apertus1p5TextConfig
from gptqmodel.models.ports.apertus.modeling_apertus1p5 import (
    Apertus1p5ForConditionalGeneration,
    Apertus1p5TextForCausalLM,
)
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


KINDS = ["apertus", "apertus1p5_text", "apertus1p5_text_pruned", "apertus1p5", "apertus1p5_pruned"]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(kind, **overrides):
    text = {"vocab_size": 96, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "num_key_value_heads": 1, "attention_dropout": 0.0,
                "pad_token_id": 3, "bos_token_id": 1, "eos_token_id": 2,
                "rope_parameters": {"rope_type": "default", "rope_theta": 12000000.0}}
    text.update(overrides)
    if kind == "apertus":
        return ApertusConfig(**text)
    if kind.endswith("_pruned"):
        text["output_vocab_size"] = 48
    if kind.startswith("apertus1p5_text"):
        return Apertus1p5TextConfig(**text)
    return Apertus1p5Config(
        text_config=text,
        vision_config={"base_channels": 32, "channel_multiplier": [1, 1], "num_res_blocks": 1,
                           "attn_resolutions": [4], "resolution": 8, "embed_dim": 8, "latent_channels": 8, "codebook_size": 8},
        audio_config={"num_filters": 4, "upsampling_ratios": [2, 2], "num_lstm_layers": 1,
                          "hidden_size": 8, "codebook_dim": 8, "codebook_size": 8},
        image_token_id=4, audio_token_id=5, image_token_offset=64, audio_token_offset=72,
    )


def tiny_model(kind, **overrides):
    cls = {"apertus": ApertusForCausalLM, "apertus1p5": Apertus1p5ForConditionalGeneration}.get(
        kind.removesuffix("_pruned"), Apertus1p5TextForCausalLM
    )
    return cls(tiny_config(kind, **overrides)).eval()


def set_attention(model, implementation):
    # Vision/audio use their fixed upstream attention, not decoder dispatch.
    if isinstance(model, Apertus1p5ForConditionalGeneration):
        model.model.language_model.set_attn_implementation(implementation)
    else:
        model.set_attn_implementation(implementation)


def tokenizer():
    special = {k: f"<{k}>" for k in (
        "image_token", "audio_token", "boi_token", "eoi_token", "image_wrapper_token",
        "eol_token", "boa_token", "eoa_token",
    )}
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3}
    vocab.update({v: i + 4 for i, v in enumerate(special.values())})
    vocab.update({f"w{i}": i for i in range(12, 96)})
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        extra_special_tokens=special,
    )


def save_processor(path):
    from gptqmodel.models.ports.apertus.feature_extraction_wavtokenizer import WavTokenizerFeatureExtractor
    from gptqmodel.models.ports.apertus.image_processing_apertus1p5 import Apertus1p5ImageProcessor
    from gptqmodel.models.ports.apertus.processing_apertus1p5 import Apertus1p5Processor

    processor = Apertus1p5Processor(
        Apertus1p5ImageProcessor(spatial_factor=2, size={"min_pixels": 64, "max_pixels": 64}),
        WavTokenizerFeatureExtractor(hop_length=4), tokenizer(),
    )
    processor.save_pretrained(path)
    return processor


def poisoned_checkpoint(path, kind):
    model = tiny_model(kind)
    model.save_pretrained(path, safe_serialization=True)
    marker = path / "EXECUTED"
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote')\n")
    payload = json.loads((path / "config.json").read_text())
    payload["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoProcessor"), "hostile.Model")
    if kind.removesuffix("_pruned") == "apertus1p5":
        for nested in ("text_config", "vision_config", "audio_config"):
            payload[nested]["auto_map"] = payload["auto_map"]
    (path / "config.json").write_text(json.dumps(payload))
    return model, marker


@pytest.mark.parametrize("model_type", ["apertus1p5", "apertus1p5_text"])
def test_apertus_15_definition_owns_modeling(model_type):
    assert isinstance(MODEL_MAP[model_type].loader, LocalModelLoader)


@pytest.mark.parametrize("kind", ["apertus", "apertus1p5_text", "apertus1p5"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_checkpoint_code_never_replaces_definition(tmp_path, monkeypatch, kind, trust, bypass):
    original, marker = poisoned_checkpoint(tmp_path, kind)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing dual-opt-in security policy is under test.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    config = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(config) is type(original.config)
    selected_trust = resolve_trust_remote_code(tmp_path, trust_remote_code=trust)
    assert selected_trust is False
    loader = MODEL_MAP[kind].loader
    instantiated = loader.from_config(config, trust_remote_code=selected_trust)
    restored = loader.from_pretrained(tmp_path, trust_remote_code=selected_trust)
    assert type(instantiated) is type(restored) is type(original)
    assert restored.state_dict().keys() == original.state_dict().keys()
    assert not marker.exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_public_load_uses_native_or_package_owned_class(tmp_path, monkeypatch, kind, trust, bypass):
    from gptqmodel import GPTQModel

    original, marker = poisoned_checkpoint(tmp_path, kind)
    if kind.removesuffix("_pruned") == "apertus1p5":
        save_processor(tmp_path)
    else:
        tokenizer().save_pretrained(tmp_path)
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Both opt-ins still cannot supersede the owned definition.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust, dtype=torch.float32)
    assert type(loaded.model) is type(original)
    if kind.removesuffix("_pruned") == "apertus1p5":
        assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
        assert loaded.load_processor().tokenizer is loaded.processor.tokenizer
    with torch.inference_mode():
        ids = torch.tensor([[1, 12, 13]])
        torch.testing.assert_close(loaded.model(ids).logits, original(ids).logits)
    assert not marker.exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_forward_cache_generation_and_safetensors(tmp_path, kind, implementation):
    torch.manual_seed(123)
    model = tiny_model(kind)
    set_attention(model, implementation)
    ids = torch.tensor([[3, 1, 12, 13, 14], [1, 15, 16, 17, 18]])
    mask = ids.ne(3).long()
    with torch.inference_mode():
        full = model(ids, attention_mask=mask, use_cache=False).logits
        prefix = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
        continuation = model(ids[:, 3:], attention_mask=mask, past_key_values=prefix.past_key_values, use_cache=True)
        torch.testing.assert_close(continuation.logits, full[:, 3:], atol=2e-6, rtol=2e-5)
        assert continuation.past_key_values.get_seq_length() == ids.shape[1]
        cached = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=True)
        uncached = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached)
        assert full.shape == (2, 5, 48 if kind.endswith("_pruned") else 96)
        assert torch.isfinite(model(ids, attention_mask=mask, labels=ids, use_cache=False).loss)
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert not list(tmp_path.glob("*.bin"))
    with safe_open(tmp_path / "model.safetensors", framework="pt") as saved:
        assert set(saved.keys()) == set(model.state_dict())
    restored = type(model).from_pretrained(tmp_path).eval()
    set_attention(restored, implementation)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids, attention_mask=mask, use_cache=False).logits, full, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("attention_bias", [False, True])
@pytest.mark.parametrize("hidden_act", ["xielu", "silu"])
def test_text_matches_exact_native_apertus_oracle(attention_bias, hidden_act):
    native = tiny_model("apertus", attention_bias=attention_bias, hidden_act=hidden_act)
    port = tiny_model("apertus1p5_text", attention_bias=attention_bias, hidden_act=hidden_act)
    port.load_state_dict(native.state_dict(), strict=True)
    ids = torch.tensor([[1, 20, 21, 22], [3, 1, 13, 24]])
    mask = ids.ne(3).long()
    with torch.inference_mode():
        expected = native(ids, attention_mask=mask, use_cache=False).logits
        actual = port(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["apertus1p5", "apertus1p5_pruned"])
def test_multimodal_image_audio_splice_cache_and_failure_retry(tmp_path, kind):
    torch.manual_seed(41)
    model = tiny_model(kind)
    pixels = torch.randn(1, 3, 8, 8)
    sizes = torch.tensor([[8, 8]])
    audio = torch.randn(1, 1, 17)
    audio_mask = torch.ones(1, 17, dtype=torch.long)
    with torch.inference_mode():
        # Encoder buffers initialize to zero, not a useful code-assignment oracle.
        # Put distinct live encoder frames into the codebook so all frames exercise
        # nontrivial nearest-neighbor selection without changing model computation.
        frames = model.model.audio_tokenizer.encoder(audio)[0].T
        codebook = model.model.audio_tokenizer.quantizer.codebook
        codebook.embed.fill_(100)
        codebook.embed[:frames.shape[0]].copy_(frames)
        codebook.embed_avg.copy_(codebook.embed)
        image_codes = model.model.get_image_tokens(pixels, sizes)
        audio_codes = model.model.get_audio_tokens(audio, audio_mask)
        assert image_codes.shape == (16,)
        assert audio_codes.shape == (5,)
        assert ((image_codes >= 64) & (image_codes < 72)).all()
        assert ((audio_codes >= 72) & (audio_codes < 80)).all()
        assert audio_codes.unique().numel() > 1
        expected_codes = torch.cdist(frames, codebook.embed).argmin(-1) + 72
        assert torch.equal(audio_codes, expected_codes)
        ids = torch.tensor([[1] + [4] * 16 + [5] * 5 + [12]])
        discrete = torch.cat((torch.tensor([1]), image_codes, audio_codes, torch.tensor([12])))[None]
        media = {"pixel_values": pixels, "image_sizes": sizes, "input_features": audio, "feature_attention_mask": audio_mask}
        reference = model(discrete, use_cache=False).logits
        actual = model(ids, **media, use_cache=True)
        torch.testing.assert_close(actual.logits, reference, rtol=0, atol=0)
        with pytest.raises((ValueError, RuntimeError), match="tokens|features|shape"):
            model(ids[:, :-2], **media, use_cache=False)
        torch.testing.assert_close(model(ids, **media, use_cache=False).logits, reference, rtol=0, atol=0)
        next_id = torch.tensor([[13]])
        extended = model(torch.cat((discrete, next_id), dim=1), use_cache=False).logits[:, -1:]
        continued = model(next_id, past_key_values=actual.past_key_values, use_cache=True).logits
        torch.testing.assert_close(continued, extended, atol=2e-6, rtol=2e-5)
        for beams in (1, 2):
            generated = model.generate(ids, **media, num_beams=beams, max_new_tokens=2, do_sample=False)
            expected = model.generate(discrete, num_beams=beams, max_new_tokens=2, do_sample=False)
            assert torch.equal(generated[:, ids.shape[1]:], expected[:, ids.shape[1]:])
        model.save_pretrained(tmp_path, safe_serialization=True)
        restored = Apertus1p5ForConditionalGeneration.from_pretrained(tmp_path).eval()
        torch.testing.assert_close(restored(ids, **media, use_cache=False).logits, reference, rtol=0, atol=0)


def test_vocab_validation_and_checkpoint_failure_retry(tmp_path):
    with pytest.raises(StrictDataclassClassValidationError, match="output_vocab_size"):
        tiny_model("apertus1p5_text", output_vocab_size=97)
    model, marker = poisoned_checkpoint(tmp_path, "apertus1p5_text")
    config = copy.deepcopy(model.config)
    config.hidden_size = 32
    with pytest.raises(RuntimeError, match="ignore_mismatched_sizes"):
        MODEL_MAP["apertus1p5_text"].loader.from_pretrained(tmp_path, config=config)
    restored = MODEL_MAP["apertus1p5_text"].loader.from_pretrained(tmp_path)
    assert restored.lm_head.weight.shape == model.lm_head.weight.shape
    assert not marker.exists()


def test_local_processor_roundtrip_ignores_checkpoint_auto_map(tmp_path):
    from gptqmodel.models.ports.apertus.processing_apertus1p5 import Apertus1p5Processor

    _, marker = poisoned_checkpoint(tmp_path, "apertus1p5")
    processor = save_processor(tmp_path)
    path = tmp_path / "processor_config.json"
    payload = json.loads(path.read_text())
    payload["auto_map"] = {"AutoProcessor": "hostile.Model"}
    path.write_text(json.dumps(payload))
    restored = Apertus1p5Processor.from_pretrained(tmp_path, trust_remote_code=True)
    assert type(restored) is type(processor)
    assert type(restored.image_processor) is type(processor.image_processor)
    assert type(restored.feature_extractor) is type(processor.feature_extractor)
    assert restored.image_processor.spatial_factor == 2
    assert restored.feature_extractor.hop_length == 4
    assert restored(text="w12 w13")["input_ids"] == processor(text="w12 w13")["input_ids"]
    inputs = restored(
        text=f"{restored.image_token} {restored.audio_token}",
        images=torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8),
        audio=[torch.linspace(-1, 1, 17).numpy()], return_tensors="pt",
    )
    assert inputs["pixel_values"].shape == (1, 3, 8, 8)
    assert inputs["feature_attention_mask"].sum() == 17
    assert (inputs["input_ids"] == restored.image_token_id).sum() == 16
    assert (inputs["input_ids"] == restored.audio_token_id).sum() == 5
    with torch.inference_mode():
        assert torch.isfinite(tiny_model("apertus1p5")(**inputs).logits).all()
    assert not marker.exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hidden_act", ["xielu", "silu"])
@pytest.mark.parametrize("attention_bias", [False, True])
def test_awq_plain_quantization_covers_every_decoder_projection_and_retry(
    monkeypatch, tmp_path, kind, hidden_act, attention_bias,
):
    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.loop_processor import CalibrationConfig
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization.config import AWQConfig

    model = tiny_model(kind, hidden_act=hidden_act, attention_bias=attention_bias)
    definition = MODEL_MAP[kind.removesuffix("_pruned")]
    prefix = "model.language_model.layers" if kind.removesuffix("_pruned") == "apertus1p5" else "model.layers"
    layer = model.get_submodule(prefix)[0]
    if kind.removesuffix("_pruned") == "apertus1p5":
        model.config.save_pretrained(tmp_path)
        save_processor(tmp_path)
    wrapper = definition(model=model, quantized=False, quantize_config=AWQConfig(device="cpu", group_size=8),
                         model_local_path=str(tmp_path))
    features, handles = {}, []
    linears = {name: module for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
    def record(name):
        def hook(module, args):
            features[f"{prefix}.0.{name}"] = args[0].detach().clone()
        return hook
    try:
        for name, module in linears.items():
            handles.append(module.register_forward_pre_hook(record(name)))
        with torch.inference_mode():
            model(torch.tensor([[3, 1, 12, 13], [1, 21, 22, 23]]), attention_mask=torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(features) == len(linears) == 6
    assert wrapper.awq_get_modules_for_scaling(layer, features, {}) == []
    processor = AWQProcessor(tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=8),
                             calibration=CalibrationConfig(), gptq_model=wrapper, model=model,
                             calculate_w_wq_diff=False)
    monkeypatch.setattr(processor, "draw_progress", lambda *a, **k: None)
    monkeypatch.setattr(processor, "log_new_row", lambda *a, **k: None)
    state = processor._get_layer_state(0)
    state.layer_module = layer
    named = {f"{prefix}.0.{name}": NamedModule(module, name, f"{prefix}.0.{name}", 0) for name, module in linears.items()}
    state.modules = dict(named)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    def fail_features(state):
        raise RuntimeError("fixture capture interrupted")
    monkeypatch.setattr(processor, "_layer_input_features", fail_features)
    with pytest.raises(RuntimeError, match="capture interrupted"):
        processor._quantize_layer(0, state)
    assert not state.quantized
    for name, value in before.items():
        torch.testing.assert_close(layer.state_dict()[name], value, rtol=0, atol=0)
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    processor._quantize_layer(0, state)
    assert state.quantized
    after = {k: v.clone() for k, v in layer.state_dict().items()}
    for name, module in named.items():
        assert {"q_scales", "q_zeros", "wq"} <= module.state.keys()
        assert torch.isfinite(module.state["q_scales"]).all()
        assert not torch.equal(before[f"{module.name}.weight"], module.weight)
    for name, value in before.items():
        if not any(name == f"{linear}.weight" for linear in linears):
            torch.testing.assert_close(after[name], value, rtol=0, atol=0)
    processor._quantize_layer(0, state)
    for name, value in after.items():
        torch.testing.assert_close(layer.state_dict()[name], value, rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 12, 13]]), use_cache=False).logits).all()
