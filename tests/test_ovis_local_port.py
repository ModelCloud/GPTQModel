# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual Ovis 1.x numerical, multimodal, security and checkpoint transitions."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.ovis import OvisQModel
from gptqmodel.models.ports.ovis.configuration_ovis import OvisConfig
from gptqmodel.models.ports.ovis.modeling_ovis import Ovis
from gptqmodel.models.ports.ovis.processing_ovis import ConversationFormatter, OvisProcessor, process_image


LAYOUTS = ["clip-llama", "clip-qwen2", "siglip-llama", "s2-gemma2", "adaptive-llama", "s2-clip-llama",
           "adaptive-clip-llama", "partition-llama", "partition-gemma2"]
PORT = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/ovis"


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(layout="partition-llama", kv_heads=2, bias=False):
    kind = layout.split("-")[-1]
    clip = "clip" in layout
    version = "1.0" if layout.startswith("clip") else ("1.6" if layout.startswith("partition") else "1.5")
    text = {"model_type": kind, "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": kv_heads, "head_dim": 8, "max_position_embeddings": 512,
            "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "attention_bias": bias, "mlp_bias": bias,
            "tie_word_embeddings": kind == "gemma2", "attn_implementation": "eager"}
    if kind == "gemma2":
        text.update(sliding_window=4, query_pre_attn_scalar=8)
    visual = {"model_type": "clip_visual_tokenizer" if clip else "siglip_visual_tokenizer", "vocab_size": 32,
              "drop_cls_token": clip, "use_indicators": version != "1.6", "hidden_stride": 2 if version == "1.6" else 1,
              "backbone_config": {"model_type": "clip_vision_model" if clip else "siglip_vision_model",
                                  "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                                  "image_size": 6, "patch_size": 2, "hidden_act": "gelu", "attention_dropout": 0.0}}
    if version == "1.5":
        visual["hd_booster"] = "s2wrapper" if layout.startswith("s2") else (
            "s2wrapper-adaptive" if layout.startswith("adaptive") else None)
    return OvisConfig(llm_config=text, visual_tokenizer_config=visual, ovis_version=version,
                      multimodal_max_length=256, attn_implementation="eager")


def tiny_model(layout="partition-llama", **kwargs):
    torch.manual_seed(587)
    return Ovis(tiny_config(layout, **kwargs)).eval()


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, **{f"token{i}": i for i in range(4, 64)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")


def inputs_for(model):
    ids = [1, -301, -300, -302, -305, 4, 5] if model.config.ovis_version == "1.6" else [1, -200, 4, 5]
    channels = 15 if model.config.visual_tokenizer_config.hd_booster else 3
    pixels = torch.linspace(-0.9, 0.8, channels * 6 * 6).reshape(1, channels, 6, 6).to(model.dtype)
    return torch.tensor([ids]), [pixels]


def write_checkpoint(path, model=None, hostile=False):
    model = model or tiny_model()
    model.save_pretrained(path)
    tiny_tokenizer().save_pretrained(path)
    marker = path / "EXECUTED_CHECKPOINT_PYTHON"
    if hostile:
        for filename, auto_map in (
            ("config.json", {"AutoConfig": "poison.Config", "AutoModelForCausalLM": "poison.Model"}),
            ("tokenizer_config.json", {"AutoTokenizer": ["poison.Tokenizer", "poison.Tokenizer"]}),
            ("preprocessor_config.json", {"AutoImageProcessor": "poison.ImageProcessor", "AutoProcessor": "poison.Processor"}),
        ):
            value = json.loads((path / filename).read_text())
            value["auto_map"] = auto_map
            (path / filename).write_text(json.dumps(value))
        (path / "processor_config.json").write_text(json.dumps({"processor_class": "PoisonProcessor", "auto_map": {"AutoProcessor": "poison.Processor"}}))
        (path / "poison.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote Python executed')\n")
        (path / "custom_generate").mkdir(exist_ok=True)
        (path / "custom_generate" / "generate.py").write_text((path / "poison.py").read_text())
    return marker


def _linear(x, module):
    return F.linear(x, module.weight.double(), None if module.bias is None else module.bias.double())


def _norm(x, module):
    return F.layer_norm(x, module.normalized_shape, module.weight.double(), module.bias.double(), module.eps)


def vision_oracle(model, pixels):
    """Independent TF4 CLIP/SigLIP encoder equations, not VisualTokenizer.encode."""
    visual = model.visual_tokenizer
    backbone = visual.backbone.vision_model
    cfg = visual.config
    batch = pixels.shape[0]
    adaptive = None
    if cfg.hd_booster:
        adaptive = torch.isinf(pixels[:, -1, -1, -1])
        pixels = pixels.masked_fill(torch.isinf(pixels), 0).reshape(batch * 5, 3, 6, 6)
    emb = backbone.embeddings
    patch = emb.patch_embedding
    x = F.conv2d(pixels.double(), patch.weight.double(), None if patch.bias is None else patch.bias.double(), stride=patch.stride)
    x = x.flatten(2).transpose(1, 2)
    if cfg.model_type == "clip_visual_tokenizer":
        x = torch.cat((emb.class_embedding.double().expand(len(x), 1, -1), x), dim=1)
    x = x + emb.position_embedding.weight.double()
    if cfg.model_type == "clip_visual_tokenizer":
        x = _norm(x, backbone.pre_layrnorm)
    for layer in backbone.encoder.layers:
        hidden = _norm(x, layer.layer_norm1)
        shape = (len(hidden), len(hidden[0]), 2, 8)
        q = _linear(hidden, layer.self_attn.q_proj).reshape(shape).transpose(1, 2)
        k = _linear(hidden, layer.self_attn.k_proj).reshape(shape).transpose(1, 2)
        v = _linear(hidden, layer.self_attn.v_proj).reshape(shape).transpose(1, 2)
        weights = (q @ k.transpose(-1, -2) / (8**0.5)).softmax(-1)
        x = x + _linear((weights @ v).transpose(1, 2).reshape_as(hidden), layer.self_attn.out_proj)
        hidden = _norm(x, layer.layer_norm2)
        x = x + _linear(F.gelu(_linear(hidden, layer.mlp.fc1)), layer.mlp.fc2)
    if cfg.drop_cls_token:
        x = x[:, 1:]
    if cfg.hd_booster:
        x = x.reshape(batch, 5, 9, 16)
        high = x[:, 1:].reshape(batch, 2, 2, 3, 3, 16).permute(0, 5, 1, 3, 2, 4).reshape(batch, 16, 6, 6)
        pooled = F.avg_pool2d(high, 2).flatten(2).transpose(1, 2)
        if cfg.hd_booster == "s2wrapper-adaptive":
            pooled[adaptive] = 0
        x = torch.cat((x[:, 0], pooled), dim=-1)
    if cfg.hidden_stride == 2:
        grid = F.pad(x.reshape(batch, 3, 3, -1), (0, 0, 0, 1, 0, 1))
        x = torch.stack([grid[:, row:row + 2, col:col + 2].reshape(batch, -1)
                         for row in (0, 2) for col in (0, 2)], dim=1)
    logits = _norm(_linear(x, visual.head[0]), visual.head[1])
    tokens = logits.softmax(-1)
    if model.config.ovis_version == "1.6":
        return F.pad(tokens, (0, 5))
    if cfg.use_indicators:
        tokens = F.pad(tokens, (0, 2))
        begin = F.one_hot(torch.full((batch, 1), 30), 32).double()
        end = F.one_hot(torch.full((batch, 1), 31), 32).double()
        tokens = torch.cat((begin, tokens, end), dim=1)
    return tokens


@pytest.mark.parametrize("layout", LAYOUTS)
def test_independent_float64_visual_vocabulary_full_decoder_and_image_merge(layout):
    model = tiny_model(layout).double()
    ids, pixels = inputs_for(model)
    tokens = vision_oracle(model, pixels[0])
    actual_tokens = model.visual_tokenizer(pixels[0])
    torch.testing.assert_close(actual_tokens, tokens, atol=2e-6, rtol=1e-5)
    visual_embeds = tokens @ model.vte.weight
    text = F.embedding(ids.clamp_min(0), model.get_wte().weight)
    if model.config.ovis_version == "1.6":
        expected = torch.cat((text[:, :1], model.vte.weight[-5:-4][None], visual_embeds,
                              model.vte.weight[-4:-3][None], model.vte.weight[-1:][None], text[:, -2:]), dim=1)
    else:
        expected = torch.cat((text[:, :1], visual_embeds, text[:, -2:]), dim=1)
    if model.config.llm_config.model_type == "gemma2":
        expected *= expected.new_tensor(32**0.5)
    _, embeds, labels, mask = model.merge_multimodal(ids, pixel_values=pixels)
    torch.testing.assert_close(embeds, expected, atol=2e-6, rtol=1e-5)
    with torch.no_grad():
        reference = model.llm(inputs_embeds=expected, attention_mask=mask, use_cache=False).logits
        actual = model(ids, pixel_values=pixels, use_cache=False).logits
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-5)
    assert (labels == -100).all()
    assert (actual - model(ids, pixel_values=[pixels[0].flip(-1)], use_cache=False).logits).abs().max() > 1e-7


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_native_cache_beam_generation_and_two_safetensors_cycles(tmp_path, layout, dtype):
    tiny_model(layout).save_pretrained(tmp_path / "initial")
    model = Ovis.from_pretrained(tmp_path / "initial", dtype=dtype, attn_implementation="eager").eval()
    ids, pixels = inputs_for(model)
    with torch.no_grad():
        first = model(ids, pixel_values=pixels, use_cache=True)
        assert first.past_key_values.get_seq_length() == first.logits.shape[1]
        continuation = torch.tensor([[7, 8, 9, 10]])
        mask = torch.ones(1, first.logits.shape[1] + 4, dtype=torch.long)
        cached = model(continuation, attention_mask=mask, past_key_values=first.past_key_values, use_cache=True)
        full = model(torch.cat((ids, continuation), -1), pixel_values=pixels, use_cache=False)
        tolerance = 8e-3 if dtype == torch.bfloat16 else (8e-4 if dtype == torch.float16 else 2e-5)
        torch.testing.assert_close(cached.logits, full.logits[:, -4:], atol=tolerance, rtol=tolerance)
        generated = [model.generate(ids, pixel_values=pixels, max_new_tokens=3, num_beams=beam, do_sample=False)
                     for beam in (1, 3)]
        expected = model(ids, pixel_values=pixels, use_cache=False).logits
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for index in range(2):
        path = tmp_path / str(index)
        model.save_pretrained(path)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        model = Ovis.from_pretrained(path, dtype=dtype, attn_implementation="eager").eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids, pixel_values=pixels, use_cache=False).logits, expected, atol=0, rtol=0)
            for beam, reference in zip((1, 3), generated):
                assert torch.equal(model.generate(ids, pixel_values=pixels, max_new_tokens=3, num_beams=beam, do_sample=False), reference)
        assert set(state) == set(model.state_dict())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kv_heads,bias", [(1, False), (2, True), (4, False)])
def test_mha_gqa_mqa_and_bias_actual_cpu_state_assignment(layout, kv_heads, bias):
    model = tiny_model(layout, kv_heads=kv_heads, bias=bias)
    ids, pixels = inputs_for(model)
    with torch.no_grad():
        reference = model(ids, pixel_values=pixels, use_cache=False).logits
    for assign in (False, True):
        fresh = Ovis(copy.deepcopy(model.config)).eval()
        fresh.load_state_dict(model.state_dict(), assign=assign, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(fresh(ids, pixel_values=pixels, use_cache=False).logits, reference, atol=0, rtol=0)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_public_model_config_tokenizer_processor_ignore_hostile_checkpoint_python(tmp_path, monkeypatch, trust, bypass, layout):
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    marker = write_checkpoint(tmp_path, tiny_model(layout), hostile=True)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(wrapper) is OvisQModel
    assert type(wrapper.model) is Ovis
    assert type(wrapper.model.config) is OvisConfig
    assert type(wrapper.processor) is OvisProcessor
    assert wrapper.model.text_tokenizer is wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
    image = Image.fromarray(np.arange(14 * 11 * 3, dtype=np.uint8).reshape(14, 11, 3))
    _, ids, pixels = wrapper.model.preprocess_inputs("<image> token4", [image], max_partition=2)
    with torch.no_grad():
        assert torch.isfinite(wrapper.model(ids[None], pixel_values=[pixels], use_cache=False).logits).all()
    assert not marker.exists()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_native_image_pixels_s2_adaptive_partition_and_failed_preprocess_retry(layout):
    model = tiny_model(layout)
    processor = OvisProcessor(model.visual_tokenizer.image_processor, tiny_tokenizer(), model.config)
    image = Image.fromarray(np.arange(17 * 11 * 3, dtype=np.uint8).reshape(11, 17, 3))
    _, ids, pixels = processor.preprocess_inputs("<image> token4", [image], max_partition=4)
    assert ids.lt(0).any()
    assert pixels.shape[-2:] == (6, 6)
    with pytest.raises(ValueError, match="placeholders"):
        processor.preprocess_inputs("<image> <image>", [image])
    _, retry_ids, retry_pixels = processor.preprocess_inputs("<image> token4", [image], max_partition=4)
    assert torch.equal(ids, retry_ids) and torch.equal(pixels, retry_pixels)
    if layout.startswith("adaptive"):
        low, _ = process_image(image.resize((6, 6)), processor.image_processor, model.config)
        assert torch.isinf(low[:, 3:]).all()
        with torch.no_grad():
            torch.testing.assert_close(model.visual_tokenizer(low), vision_oracle(model, low).float(), atol=2e-6, rtol=1e-5)
    if layout.startswith("partition"):
        assert (ids == -300).sum() == pixels.shape[0]
    with torch.no_grad():
        assert torch.isfinite(model(ids[None], pixel_values=[pixels], use_cache=False).logits).all()


class CharacterTokenizer:
    def __call__(self, value, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=[ord(char) for char in value])


@pytest.mark.parametrize("kind,version", [("Llama3", "1.6"), ("Gemma", "1.5"), ("Qwen", "1.0"), ("Qwen", "1.5")])
def test_source_exact_formatter_protocols_and_labels(kind, version):
    formatter = ConversationFormatter(CharacterTokenizer(), kind + "ConversationFormatter", version)
    conversations = [{"from": "human", "value": " <image> A "}, {"from": "gpt", "value": " B "}]
    original = copy.deepcopy(conversations)
    prompt, ids, labels = formatter.format(conversations)
    assert conversations == original
    if kind == "Llama3":
        expected = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful and honest multimodal assistant.<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n<image> A<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nB<|eot_id|>")
        learned = "B<|eot_id|>"
    elif kind == "Gemma":
        expected = "<bos><start_of_turn>user\n<image> A<end_of_turn>\n<start_of_turn>model\nB<end_of_turn>\n"
        learned = "B<end_of_turn>"
    else:
        expected = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n <image> A <|im_end|>\n<|im_start|>assistant\n B <|im_end|>"
        if version != "1.0":
            expected += "\n"
        learned = " B <|im_end|>"
    assert prompt == expected
    expected_ids = [ord(char) for char in expected.replace("<image>", "\x00")]
    expected_ids[expected_ids.index(0)] = -200
    assert ids == expected_ids
    assert [value for value in labels if value != -100] == [ord(char) for char in learned]


@pytest.mark.parametrize("layout,policy", [
    ("clip-qwen2", {"bos_token_id": 151643, "do_sample": True, "eos_token_id": [151645, 151643],
                   "multimodal_max_length": 2048, "pad_token_id": 151643, "repetition_penalty": 1.05,
                   "temperature": 0.7, "top_k": 20, "top_p": 0.8}),
    ("partition-llama", {"bos_token_id": 128000, "do_sample": True, "eos_token_id": [128001, 128008, 128009],
                         "multimodal_max_length": 2624, "temperature": 0.6, "top_p": 0.9}),
    ("partition-gemma2", {"_from_model_config": True, "bos_token_id": 2, "cache_implementation": "hybrid",
                          "eos_token_id": [1, 107], "pad_token_id": 0}),
])
def test_released_generation_policy_public_load_and_two_safetensors_cycles(tmp_path, layout, policy):
    path = tmp_path / "initial"
    marker = write_checkpoint(path, tiny_model(layout), hostile=True)
    (path / "generation_config.json").write_text(json.dumps(policy))
    expected = None
    for index in range(3):
        wrapper = GPTQModel.load(str(path), device="cpu", dtype=torch.float32, trust_remote_code=True)
        generation = wrapper.model.generation_config
        for name, value in policy.items():
            assert getattr(generation, name) == value
        assert wrapper.model.can_generate()
        ids, pixels = inputs_for(wrapper.model)
        before = generation.to_dict()
        actual = wrapper.model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=2,
                                        do_sample=False, eos_token_id=None, pad_token_id=0)
        assert generation.to_dict() == before
        if expected is None:
            expected = actual
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert not marker.exists()
        if index < 2:
            path = tmp_path / f"round-{index}"
            wrapper.model.save_pretrained(path)
            saved = json.loads((path / "generation_config.json").read_text())
            assert all(saved.get(name) == value for name, value in policy.items() if name != "_from_model_config")
            assert list(path.glob("*.safetensors")) and not list(path.rglob("*.bin"))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_version_roundtrip_and_legacy_raw_metadata_selection(layout):
    original = tiny_config(layout)
    saved = original.to_dict()
    assert OvisConfig(**saved).ovis_version == original.ovis_version
    saved.pop("ovis_version")
    if original.ovis_version != "1.5":
        saved["visual_tokenizer_config"].pop("hd_booster")
    assert OvisConfig(**saved).ovis_version == original.ovis_version


@pytest.mark.parametrize("layout", ["clip-qwen2", "siglip-llama", "s2-clip-llama", "partition-gemma2"])
def test_ragged_multiple_images_text_only_rows_masks_labels_and_generation(layout):
    model = tiny_model(layout)
    row, pixels = inputs_for(model)
    row = torch.cat((row, row[:, 1:]), dim=1)
    ids = F.pad(row, (2, 0), value=0).expand(2, -1).clone()
    ids[1] = 0
    ids[1, -3:] = torch.tensor([1, 4, 5])
    mask = ids.ne(0)
    groups = [pixels[0].expand(2, -1, -1, -1), None]
    targets = ids.clone()
    _, embeds, labels, merged_mask = model.merge_multimodal(ids, mask, targets, groups, left_padding=True)
    assert not merged_mask[1, :-3].any() and merged_mask[1, -3:].all()
    torch.testing.assert_close(embeds[1, -3:], model._decoder_embeddings(model._raw_text_embeddings(ids[1, -3:])))
    _, one, one_labels, one_mask = model.merge_multimodal(row, text_labels=row, pixel_values=[groups[0]], left_padding=True)
    torch.testing.assert_close(embeds[0, 2:], one[0])
    assert torch.equal(labels[0, 2:], one_labels[0])
    assert torch.equal(merged_mask[0, 2:], one_mask[0])
    with torch.no_grad():
        result = model.generate(ids, attention_mask=mask, pixel_values=groups, do_sample=False, num_beams=2,
                                max_new_tokens=2, min_new_tokens=2, eos_token_id=None)
        assert result.shape == (2, 2)
        fresh = model.generate(ids, attention_mask=mask, pixel_values=groups, do_sample=False, num_beams=2,
                               max_new_tokens=2, min_new_tokens=2, eos_token_id=None)
    assert torch.equal(result, fresh)


@pytest.mark.parametrize("mode", ["softmax", "st_argmax", "gumbel_argmax"])
@pytest.mark.parametrize("indicators", [False, True])
def test_legacy_visual_stride_three_and_tokenization_modes(mode, indicators):
    config = tiny_config("siglip-llama")
    config.visual_tokenizer_config.hidden_stride = 3
    config.visual_tokenizer_config.use_indicators = indicators
    config.visual_tokenizer_config.tokenize_function = mode
    model = Ovis(config).eval()
    pixels = torch.linspace(-1, 1, 108).reshape(1, 3, 6, 6)
    output = model.visual_tokenizer(pixels)
    assert output.shape == (1, 3 if indicators else 1, 32)
    torch.testing.assert_close(output.sum(-1), torch.ones_like(output[..., 0]))
    if mode != "softmax":
        assert ((output == 0) | (output == 1)).all()
    if indicators:
        assert output[0, 0, -2] == 1 and output[0, -1, -1] == 1


@pytest.mark.parametrize("asset", ["model.safetensors", "preprocessor_config.json", "tokenizer.json"])
def test_public_missing_asset_failure_then_repaired_same_path(tmp_path, asset):
    write_checkpoint(tmp_path)
    file = tmp_path / asset
    original = file.read_bytes()
    file.unlink()
    with pytest.raises((OSError, ValueError)):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    file.write_bytes(original)
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    ids, pixels = inputs_for(wrapper.model)
    with torch.no_grad():
        assert torch.isfinite(wrapper.model(ids, pixel_values=pixels, use_cache=False).logits).all()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_native_attention_backend_capability_and_numerical_parity(tmp_path, layout):
    source = tiny_model(layout)
    source.save_pretrained(tmp_path)
    if source.config.llm_config.model_type == "gemma2":
        with pytest.raises(ValueError, match="does not support"):
            Ovis.from_pretrained(tmp_path, attn_implementation="sdpa")
        assert not source._supports_sdpa
        return
    sdpa = Ovis.from_pretrained(tmp_path, attn_implementation="sdpa").eval()
    ids, pixels = inputs_for(source)
    assert sdpa.llm.config._attn_implementation == "sdpa"
    assert sdpa.visual_tokenizer.backbone.vision_model.config._attn_implementation == "sdpa"
    with torch.no_grad():
        expected = source(ids, pixel_values=pixels, use_cache=False).logits
        actual = sdpa(ids, pixel_values=pixels, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("layout", ["clip-qwen2", "s2-gemma2", "adaptive-llama", "partition-llama"])
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_calibrated_plain_awq_pack_infer_and_two_exact_reload_cycles(tmp_path, layout, offload):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
    source = tiny_model(layout).half()
    write_checkpoint(tmp_path, source)
    config = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=config, backend=BACKEND.TORCH, dtype=torch.float16)
    ids, pixels = inputs_for(source)
    image = Image.fromarray(np.arange(11 * 13 * 3, dtype=np.uint8).reshape(11, 13, 3))
    calibration = [{"image": image, "conversations": [{"from": "human", "value": "<image> token4 token5 token6"},
                                                       {"from": "gpt", "value": "token7 token8"}]} for _ in range(2)]
    dense = {name: value.clone() for name, value in source.state_dict().items()
             if not name.startswith("llm.model.layers.") or "norm" in name}
    prepared = wrapper.preprocess_dataset(calibration[0])
    calibration_ids = prepared["input_ids"]
    image_embeddings = vision_oracle(source, prepared["pixel_values"].half()) @ source.vte.weight.double()
    atom = -300 if source.config.ovis_version == "1.6" else -200
    pieces = []
    image_index = 0
    for token in calibration_ids.tolist():
        if token == atom:
            pieces.append(image_embeddings[image_index])
            image_index += 1
        elif token < 0:
            pieces.append(source.vte.weight.double()[[-token - 301 - 5]])
        else:
            pieces.append(source.get_wte().weight.double()[[token]])
    expected_capture = torch.cat(pieces)[None]
    if source.config.llm_config.model_type == "gemma2":
        expected_capture *= 32**0.5
    captures = []

    def capture(_module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        if not captures:
            captures.append(hidden.detach().cpu().clone())

    handle = wrapper.model.llm.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    for layer in wrapper.model.llm.model.layers:
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    try:
        wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    finally:
        handle.remove()
    assert captures
    torch.testing.assert_close(captures[0].double(), expected_capture, atol=3e-3, rtol=1e-2)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert len(packed) == 14
    assert all(name.startswith("llm.model.layers.") for name in packed)
    for name, value in dense.items():
        torch.testing.assert_close(wrapper.model.state_dict()[name], value, atol=0, rtol=0)
    with torch.no_grad():
        expected = wrapper.model(ids, pixel_values=pixels, use_cache=False).logits
    state = {key: value.clone() for key, value in wrapper.model.state_dict().items()}
    assert torch.isfinite(expected).all()
    for index in range(2):
        saved = tmp_path / f"quantized-{index}"
        wrapper.save_quantized(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.rglob("*.bin"))
        wrapper = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 14
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(ids, pixel_values=pixels, use_cache=False).logits, expected, atol=0, rtol=0)
        assert set(state) == set(wrapper.model.state_dict())
        for name, value in wrapper.model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
