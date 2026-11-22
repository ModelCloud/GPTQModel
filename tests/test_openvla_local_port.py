# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual OpenVLA computation, released ownership, actions and public consumers."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.openvla import OpenVLAQModel
from gptqmodel.models.ports.openvla.configuration_prismatic import (
    VISION_BACKBONE_TO_TIMM_ID,
    OpenVLAConfig,
)
from gptqmodel.models.ports.openvla.modeling_prismatic import OpenVLAForActionPrediction
from gptqmodel.models.ports.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from gptqmodel.quantization.awq.utils.module import get_op_name
from tests.port_assets import read_port_asset


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/openvla/fixtures"
NORMS = {"robot": {"action": {"q01": [-2.0, 1.0, 9.0], "q99": [4.0, 5.0, 13.0],
                              "mask": [True, True, False]}}}
TOKEN_IDS = [1, 512, 29901, 1724, 3158, 881, 278, 19964, 2125, 304, 5839, 701, 278,
             18002, 29973, 13, 3744, 29901, 29871]
PROMPT = "In: What action should the robot take to pick up the cup?\nOut: "


@pytest.fixture(autouse=True, scope="module")
def _bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _asset(name):
    return read_port_asset(FIXTURES / name)


def _config(fused=True, backbone=None, llm="llama2-7b-pure", **kwargs):
    backbone = backbone or ("dinosiglip-vit-so-224px" if fused else "siglip-vit-so400m")
    count = len(VISION_BACKBONE_TO_TIMM_ID[backbone])
    text = {"vocab_size": 257, "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4,
                "num_key_value_heads": 2, "num_hidden_layers": 2, "max_position_embeddings": 128,
                "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": None, "attention_dropout": 0.0}
    text.update(kwargs.pop("text_config", {}))
    if llm == "phi-2-3b":
        text.update(partial_rotary_factor=0.5, resid_pdrop=0.0, embd_pdrop=0.0)
    values = {"vision_backbone_id": backbone, "llm_backbone_id": llm, "pad_token_id": 0,
                  "image_sizes": [32 if backbone == "in1k-vit-l" else 28] * count,
                  "vision_backbone_kwargs": [{"embed_dim": 16 + 8 * i, "depth": 3, "num_heads": 4, "mlp_ratio": 2}
                                          for i in range(count)], "text_config": text, "norm_stats": copy.deepcopy(NORMS),
                  "attn_implementation": "eager"}
    values.update(kwargs)
    return OpenVLAConfig(**values)


def _model(dtype=torch.float32, **kwargs):
    torch.manual_seed(826)
    return OpenVLAForActionPrediction._from_config(_config(**kwargs), dtype=dtype).eval()


def _pixels(config, batch=1, dtype=torch.float32):
    channels = 6 if config.use_fused_vision_backbone else 3
    size = config.image_sizes[0]
    return torch.linspace(-1.7, 2.3, batch * channels * size * size, dtype=dtype).reshape(batch, channels, size, size)


def _write_assets(path, config):
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                 "processor_config.json"):
        (path / name).write_bytes(_asset(name))
    count = len(config.image_sizes)
    PrismaticImageProcessor(
        use_fused_vision_backbone=config.use_fused_vision_backbone, image_resize_strategy="resize-naive",
        input_sizes=[[3, s, s] for s in config.image_sizes], means=[[0.485, 0.456, 0.406]] * count,
        stds=[[0.229, 0.224, 0.225]] * count,
    ).save_pretrained(path)


def _vision_equations(model, pixels):
    """Independent float64 ViT equations from pinned timm/OpenVLA state tensors."""
    weights = {k: v.double() for k, v in model.state_dict().items()}

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name, eps):
        return F.layer_norm(x, (x.shape[-1],), weights[name + ".weight"], weights[name + ".bias"], eps)

    x = F.conv2d(pixels.double(), weights["patch_embed.proj.weight"], weights.get("patch_embed.proj.bias"),
                 stride=model.patch_embed.patch_size).flatten(2).transpose(1, 2)
    if model.no_embed_class:
        x = x + weights["pos_embed"]
    prefix = [weights[k].expand(x.shape[0], -1, -1) for k in ("cls_token", "reg_token") if k in weights]
    x = torch.cat((*prefix, x), 1) if prefix else x
    if not model.no_embed_class:
        x = x + weights["pos_embed"]
    if "norm_pre.weight" in weights:
        x = norm(x, "norm_pre", model.norm_pre.eps)
    # The final block and output norm are checkpoint-owned but deliberately not executed.
    for i in range(len(model.blocks) - 1):
        p = f"blocks.{i}."
        block = model.blocks[i]
        hidden = norm(x, p + "norm1", block.norm1.eps)
        batch, length, width = hidden.shape
        q, k, v = linear(hidden, p + "attn.qkv").reshape(
            batch, length, 3, block.attn.num_heads, -1,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        probabilities = ((q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)).softmax(-1)
        attention = linear((probabilities @ v).transpose(1, 2).reshape(batch, length, width), p + "attn.proj")
        if p + "ls1.scale_factor" in weights:
            attention = attention * weights[p + "ls1.scale_factor"]
        x = x + attention
        hidden = linear(norm(x, p + "norm2", block.norm2.eps), p + "mlp.fc1")
        if type(block.mlp.act).__name__ == "QuickGELU":
            hidden = hidden * torch.sigmoid(1.702 * hidden)
        else:
            hidden = F.gelu(hidden, approximate=getattr(block.mlp.act, "approximate", "none"))
        hidden = linear(hidden, p + "mlp.fc2")
        if p + "ls2.scale_factor" in weights:
            hidden = hidden * weights[p + "ls2.scale_factor"]
        x = x + hidden
    return x[:, model.num_prefix_tokens:]


def _projector_equations(model, patches):
    w = {k: v.double() for k, v in model.state_dict().items()}
    hidden = F.linear(F.gelu(F.linear(patches, w["fc1.weight"], w["fc1.bias"])),
                      w["fc2.weight"], w["fc2.bias"])
    if model.use_fused_vision_backbone:
        hidden = F.linear(F.gelu(hidden), w["fc3.weight"], w["fc3.bias"])
    return hidden


def _image_equations(model, pixels):
    pieces = [_vision_equations(model.vision_backbone.featurizer, pixels[:, :3])]
    if model.config.use_fused_vision_backbone:
        pieces.append(_vision_equations(model.vision_backbone.fused_featurizer, pixels[:, 3:]))
    return _projector_equations(model.projector, torch.cat(pieces, -1))


def _llama_equations(model, embeddings, padding):
    """Full grouped-query Llama equations; no port/native forward/cache helpers."""
    c = model.config
    w = {k: v.double() for k, v in model.state_dict().items()}
    x = embeddings.double()
    batch, length, width = x.shape
    heads, kv = c.num_attention_heads, c.num_key_value_heads
    dim = c.head_dim

    def linear(hidden, name):
        return F.linear(hidden, w[name + ".weight"], w.get(name + ".bias"))

    def norm(hidden, name):
        return hidden * (hidden.square().mean(-1, keepdim=True) + c.rms_norm_eps).rsqrt() * w[name + ".weight"]

    positions = torch.arange(length, dtype=torch.float64)
    inverse = 10000.0 ** (-torch.arange(0, dim, 2).double() / dim)
    theta = positions[:, None] * inverse
    cosine, sine = theta.cos()[None, None], theta.sin()[None, None]

    def rotate(hidden):
        left, right = hidden.chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1)

    allowed = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    allowed = allowed[None, None] & padding[:, None, None, :].bool()
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        hidden = norm(x, p + "input_layernorm")
        q = linear(hidden, p + "self_attn.q_proj").reshape(batch, length, heads, dim).transpose(1, 2)
        k = linear(hidden, p + "self_attn.k_proj").reshape(batch, length, kv, dim).transpose(1, 2)
        v = linear(hidden, p + "self_attn.v_proj").reshape(batch, length, kv, dim).transpose(1, 2)
        k = rotate(k).repeat_interleave(heads // kv, dim=1)
        v = v.repeat_interleave(heads // kv, dim=1)
        scores = ((rotate(q) @ k.transpose(-1, -2)) / (dim ** 0.5)).masked_fill(~allowed, -torch.inf)
        attention = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + linear(attention, p + "self_attn.o_proj")
        hidden = norm(x, p + "post_attention_layernorm")
        hidden = F.silu(linear(hidden, p + "mlp.gate_proj")) * linear(hidden, p + "mlp.up_proj")
        x = x + linear(hidden, p + "mlp.down_proj")
    return linear(norm(x, "model.norm"), "lm_head")


@pytest.mark.parametrize("backbone", list(VISION_BACKBONE_TO_TIMM_ID))
def test_openvla_real_vision_layouts_match_independent_equations(backbone):
    model = _model(backbone=backbone)
    with torch.no_grad():
        for name, parameter in model.vision_backbone.named_parameters():
            if name.endswith("scale_factor"):
                parameter.copy_(torch.linspace(0.2, 1.1, parameter.numel()).reshape_as(parameter))
    pixels = _pixels(model.config)
    with torch.inference_mode():
        actual = model.projector(model.vision_backbone(pixels))
        expected = _image_equations(model, pixels)
        torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=4e-5)
        altered = pixels.flip(-1).clone()
        altered[:, :, ::2] += 0.7
        assert (actual - model.projector(model.vision_backbone(altered))).abs().max() > 1e-6


@pytest.mark.parametrize("fused,kv_heads,bias", [(False, 1, False), (True, 2, True), (True, 4, False)])
def test_openvla_multimodal_llama_matches_independent_equations(fused, kv_heads, bias):
    model = _model(fused=fused, text_config={"num_key_value_heads": kv_heads, "attention_bias": bias, "mlp_bias": bias})
    ids = torch.tensor([[1, 7, 9, 4], [1, 6, 8, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    pixels = _pixels(model.config, batch=2)
    with torch.inference_mode():
        image_features = _image_equations(model, pixels)
        embeddings = F.embedding(ids, model.get_input_embeddings().weight.double())
        expanded = torch.cat((embeddings[:, :1], image_features, embeddings[:, 1:]), 1)
        expanded_mask = torch.cat((mask[:, :1], torch.ones(2, 4), mask[:, 1:]), 1)
        expected = _llama_equations(model.language_model, expanded, expanded_mask)
        actual = model(ids, pixel_values=pixels, attention_mask=mask, output_projector_features=True)
        torch.testing.assert_close(actual.projector_features.double(), image_features, atol=1e-7, rtol=4e-5)
        torch.testing.assert_close(actual.logits.double(), expected, atol=3e-7, rtol=5e-5)


@pytest.mark.parametrize("fused,dtype,implementation", [
    (False, torch.float32, "eager"), (True, torch.bfloat16, "sdpa"), (True, torch.float16, "eager"),
])
def test_openvla_cache_generation_safetensors_and_assignment(tmp_path, fused, dtype, implementation):
    model = _model(fused=fused, attn_implementation=implementation, dtype=dtype)
    ids = torch.tensor([[1, 7, 9, 4, 6, 8]])
    mask = torch.ones_like(ids)
    pixels = _pixels(model.config, dtype=dtype)
    atol = 4e-7 if dtype == torch.float32 else 0.008
    with torch.inference_mode():
        expected = model(ids, pixel_values=pixels, attention_mask=mask).logits
        prefix = model(ids[:, :1], pixel_values=pixels, attention_mask=mask[:, :1], use_cache=True)
        assert prefix.past_key_values.get_seq_length() == 5
        second = model(ids[:, 1:5], past_key_values=prefix.past_key_values,
                       attention_mask=mask[:, :5], use_cache=True)
        assert second.past_key_values.get_seq_length() == 9
        third = model(ids[:, 5:], past_key_values=second.past_key_values,
                      attention_mask=mask, use_cache=True)
        torch.testing.assert_close(second.logits, expected[:, 5:9], atol=atol, rtol=atol)
        torch.testing.assert_close(third.logits, expected[:, 9:], atol=atol, rtol=atol)
        for beams in (1, 3):
            generated = model.generate(ids[:, :3], pixel_values=pixels, max_new_tokens=3,
                                       num_beams=beams, do_sample=False, eos_token_id=None)
            assert generated.shape == (1, 6)
            if beams == 1:
                oracle = ids[:, :3].clone()
                for _ in range(3):
                    next_id = model(oracle, pixel_values=pixels, use_cache=False).logits[:, -1].argmax(-1)
                    oracle = torch.cat((oracle, next_id[:, None]), 1)
                assert torch.equal(generated, oracle)
        for assign in (False, True):
            target = _model(fused=fused, attn_implementation=implementation, dtype=dtype)
            target.load_state_dict(model.state_dict(), assign=assign)
            torch.testing.assert_close(target(ids, pixel_values=pixels).logits, expected, atol=0, rtol=0)
            del target
        for cycle in range(2):
            path = tmp_path / str(cycle)
            model.save_pretrained(path, max_shard_size="1MB")
            assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
            loaded = OpenVLAQModel.loader.from_pretrained(path, dtype=dtype, attn_implementation=implementation)
            for key, value in model.state_dict().items():
                assert torch.equal(value, loaded.state_dict()[key]), key
            torch.testing.assert_close(loaded(ids, pixel_values=pixels).logits, expected, atol=0, rtol=0)
            model = loaded


@pytest.mark.parametrize("llm", ["mistral-v0.1-7b-instruct", "phi-2-3b", "vicuna-v15-13b"])
def test_openvla_source_language_variants_preserve_native_decoder(llm):
    model = _model(llm=llm)
    pixels = _pixels(model.config)
    ids = torch.tensor([[1, 7, 9, 4]])
    with torch.inference_mode():
        patches = _image_equations(model, pixels).float()
        text = model.get_input_embeddings()(ids)
        expanded = torch.cat((text[:, :1], patches, text[:, 1:]), 1)
        expected = model.language_model(inputs_embeds=expanded).logits
        actual = model(ids, pixel_values=pixels).logits
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        generated = model.generate(ids, pixel_values=pixels, max_new_tokens=2, eos_token_id=None)
        assert generated.shape == (1, 6)


def test_openvla_labels_tuple_outputs_causality_and_failure_retry():
    model = _model()
    ids = torch.tensor([[1, 7, 9, 4]])
    pixels = _pixels(model.config)
    with torch.inference_mode():
        output = model(ids, pixel_values=pixels, labels=ids, output_hidden_states=True,
                       output_attentions=True, output_projector_features=True, use_cache=False)
        labels = torch.cat((ids[:, :1], ids.new_full((1, 4), -100), ids[:, 1:]), 1)
        loss = F.cross_entropy(output.logits[:, :-1].reshape(-1, 257), labels[:, 1:].reshape(-1))
        torch.testing.assert_close(output.loss, loss)
        assert len(output.hidden_states) == 3 and len(output.attentions) == 2
        tuple_output = model(ids, pixel_values=pixels, use_cache=False, return_dict=False)
        torch.testing.assert_close(tuple_output[0], output.logits)
        changed = ids.clone()
        changed[0, -1] = 11
        torch.testing.assert_close(model(changed, pixel_values=pixels).logits[:, :-1], output.logits[:, :-1])
        with pytest.raises(ValueError, match="same number"):
            model(ids, pixel_values=pixels.expand(2, -1, -1, -1))
        with pytest.raises(ValueError, match="attention mask"):
            model(ids, pixel_values=pixels, attention_mask=torch.ones(1, 3))
        with pytest.raises(ValueError, match="DynamicCache"):
            model(ids, past_key_values=((torch.zeros(1), torch.zeros(1)),))
        torch.testing.assert_close(model(ids, pixel_values=pixels).logits, output.logits)
        cached = model(ids[:, :2], pixel_values=pixels).past_key_values
        before = cached.get_seq_length()
        with pytest.raises(ValueError, match="cached attention mask"):
            model(ids[:, 2:], past_key_values=cached, attention_mask=torch.ones(1, 3))
        assert cached.get_seq_length() == before
        retry = model(ids[:, 2:], past_key_values=cached, attention_mask=torch.ones_like(ids))
        torch.testing.assert_close(retry.logits, output.logits[:, -2:], atol=3e-7, rtol=2e-5)


@pytest.mark.parametrize("trust,bypass", [(None, None), (False, "1"), (True, "0")])
def test_openvla_public_hostile_model_tokenizer_processor_and_retry(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = _model(text_config={"vocab_size": 32064})
    model.save_pretrained(tmp_path)
    _write_assets(tmp_path, model.config)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "checkpoint_code.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python executed')\n"
    )
    for filename in ("config.json", "tokenizer_config.json", "preprocessor_config.json", "processor_config.json"):
        payload = json.loads((tmp_path / filename).read_text())
        payload["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoProcessor", "AutoImageProcessor"), "checkpoint_code.Evil")
        payload["auto_map"]["AutoTokenizer"] = ["checkpoint_code.Evil", "checkpoint_code.Evil"]
        if filename == "config.json":
            payload["text_config"]["auto_map"] = {"AutoModelForCausalLM": "checkpoint_code.Evil"}
        (tmp_path / filename).write_text(json.dumps(payload))
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                           attn_implementation="eager")
    assert isinstance(loaded, OpenVLAQModel)
    assert type(loaded.model) is OpenVLAForActionPrediction
    assert type(loaded.model.config) is OpenVLAConfig
    assert type(loaded.processor) is PrismaticProcessor
    assert type(loaded.processor.image_processor) is PrismaticImageProcessor
    assert loaded.trust_remote_code is False and not marker.exists()
    image = Image.fromarray(np.arange(17 * 23 * 3, dtype=np.uint8).reshape(17, 23, 3))
    inputs = loaded.processor(PROMPT, image)
    assert inputs["input_ids"].tolist() == [TOKEN_IDS]
    assert inputs["pixel_values"].shape == (1, 6, 28, 28)
    with torch.inference_mode():
        expected = model(**inputs).logits
        torch.testing.assert_close(loaded.model(**inputs).logits, expected, atol=0, rtol=0)
        assert loaded.model.generate(**inputs, max_new_tokens=2).shape == (1, len(TOKEN_IDS) + 2)
    broken = (tmp_path / "preprocessor_config.json").read_bytes()
    (tmp_path / "preprocessor_config.json").write_text('{"input_sizes":[[3,-1,-1]]}')
    with pytest.raises(ValueError, match="positive square"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    (tmp_path / "preprocessor_config.json").write_bytes(broken)
    retry = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust,
                          attn_implementation="eager")
    with torch.inference_mode():
        torch.testing.assert_close(retry.model(**inputs).logits, expected, atol=0, rtol=0)
    saved = tmp_path / "saved"
    retry.model.save_pretrained(saved)
    retry.processor.save_pretrained(saved)
    reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert reloaded.processor(PROMPT, image)["input_ids"].tolist() == [TOKEN_IDS]
    torch.testing.assert_close(reloaded.processor(PROMPT, image)["pixel_values"], inputs["pixel_values"])
    assert not marker.exists()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("strategy", ["letterbox", "resize-naive", "resize-crop"])
def test_openvla_image_processor_matches_independent_pil_equations(tmp_path, fused, strategy):
    from PIL import ImageOps

    count = 2 if fused else 1
    means = [[0.485, 0.456, 0.406], [0.5, 0.5, 0.5]][:count]
    stds = [[0.229, 0.224, 0.225], [0.5, 0.5, 0.5]][:count]
    processor = PrismaticImageProcessor(use_fused_vision_backbone=fused, image_resize_strategy=strategy,
                                        input_sizes=[[3, 28, 28]] * count,
                                        means=means, stds=stds)
    image = Image.fromarray(np.arange(18 * 31 * 3, dtype=np.uint8).reshape(18, 31, 3))
    expected_image = image
    if strategy == "letterbox":
        expected_image = ImageOps.expand(image, border=(0, 6, 0, 6),
                                        fill=tuple(int(x * 255) for x in means[-1]))
    width, height = expected_image.size
    if strategy == "resize-naive":
        size = (28, 28)
    else:
        size = (int(28 * width / height), 28) if width > height else (28, int(28 * height / width))
    resized = expected_image.resize(size, resample=Image.Resampling.BICUBIC)
    left, top = round((size[0] - 28) / 2), round((size[1] - 28) / 2)
    resized = resized.crop((left, top, left + 28, top + 28))
    pixels = np.asarray(resized).transpose(2, 0, 1).astype(np.float32) / 255
    expected = np.concatenate([
        (pixels - np.array(mean, dtype=np.float32)[:, None, None]) / np.array(std, dtype=np.float32)[:, None, None]
        for mean, std in zip(means, stds)
    ])
    actual = processor(image, return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(actual[0], torch.from_numpy(expected), atol=0, rtol=0)
    processor.save_pretrained(tmp_path)
    reloaded = PrismaticImageProcessor.from_pretrained(tmp_path)
    torch.testing.assert_close(actual, reloaded(image, return_tensors="pt")["pixel_values"], atol=0, rtol=0)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("beams", [1, 3])
def test_openvla_batched_left_padding_and_beams_match_independent_image_prefill(fused, beams):
    model = _model(fused=fused)
    ids = torch.tensor([[1, 7, 9, 4], [0, 0, 1, 9]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    pixels = _pixels(model.config, batch=2)
    with torch.inference_mode():
        patches = _image_equations(model, pixels).float()
        text = F.embedding(ids, model.get_input_embeddings().weight)
        embeddings = torch.cat((text[:, :1], patches, text[:, 1:]), 1)
        expanded_mask = torch.cat((mask[:, :1], torch.ones(2, 4, dtype=torch.long), mask[:, 1:]), 1)
        expected = model.language_model.generate(
            input_ids=ids, inputs_embeds=embeddings, attention_mask=expanded_mask,
            max_new_tokens=3, num_beams=beams, do_sample=False, eos_token_id=None,
            return_dict_in_generate=True, output_scores=True,
        )
        actual = model.generate(ids, pixel_values=pixels, attention_mask=mask, max_new_tokens=3,
                                num_beams=beams, do_sample=False, eos_token_id=None,
                                return_dict_in_generate=True, output_scores=True)
        assert torch.equal(actual.sequences, expected.sequences)
        for score, reference in zip(actual.scores, expected.scores):
            torch.testing.assert_close(score, reference, atol=5e-7, rtol=1e-5)


def test_openvla_public_corrupt_tokenizer_repair_retry_and_source_immutability(tmp_path):
    from gptqmodel import GPTQModel

    model = _model(text_config={"vocab_size": 32064})
    model.save_pretrained(tmp_path)
    _write_assets(tmp_path, model.config)
    pristine = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    (tmp_path / "tokenizer.json").write_text('{"model": "broken"}')
    with pytest.raises(Exception, match="ModelUntagged"):
        GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
    (tmp_path / "tokenizer.json").write_bytes(pristine["tokenizer.json"])
    for _ in range(2):
        loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32)
        assert loaded.processor(PROMPT)["input_ids"].tolist() == [TOKEN_IDS]
        for name, value in pristine.items():
            assert (tmp_path / name).read_bytes() == value, name


def test_openvla_released_tokenizer_and_asset_hashes(tmp_path):
    from tokenizers import Tokenizer

    assert hashlib.sha256(_asset("tokenizer.json")).hexdigest() == "8f5e2869e1807b8bb3c7717a294539c37e9de5d728ad60e31ef57e83cc5ea527"
    backend = Tokenizer.from_str(_asset("tokenizer.json").decode())
    assert backend.encode(PROMPT).ids == TOKEN_IDS
    assert backend.encode("Hello 世界 café").ids == [1, 15043, 29871, 30793, 30967, 274, 28059]
    for name in ("config.json", "preprocessor_config.json", "processor_config.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        (tmp_path / name).write_bytes(_asset(name))
    processor = PrismaticProcessor.from_pretrained(tmp_path)
    assert processor(PROMPT)["input_ids"].tolist() == [TOKEN_IDS]
    assert processor("Hello 世界 café")["input_ids"].tolist() == [[1, 15043, 29871, 30793, 30967, 274, 28059]]


def test_openvla_generation_cache_retains_visual_mask_coordinates():
    model = _model()
    ids = torch.tensor([[1, 7, 9, 4]])
    pixels = _pixels(model.config)
    with torch.inference_mode():
        generated = model.generate(ids, pixel_values=pixels, max_new_tokens=2, return_dict_in_generate=True)
        assert generated.past_key_values.get_seq_length() == 9
        last = model(generated.sequences[:, -1:], past_key_values=generated.past_key_values,
                     attention_mask=torch.ones_like(generated.sequences)).logits
        expected = model(generated.sequences, pixel_values=pixels, use_cache=False).logits[:, -1:]
        torch.testing.assert_close(last, expected, atol=3e-7, rtol=2e-5)
        text_cache = model(ids, use_cache=True).past_key_values
        assert getattr(text_cache, "prismatic_patch_count", 0) == 0


@pytest.mark.parametrize("dtype,trailing_empty", [(torch.float32, False), (torch.bfloat16, True), (torch.float16, False)])
def test_openvla_action_bins_prompt_mask_and_real_generation(dtype, trailing_empty):
    from transformers import LogitsProcessor, LogitsProcessorList

    model = _model(dtype=dtype, text_config={"vocab_size": 32064})
    ids = torch.tensor([TOKEN_IDS if trailing_empty else TOKEN_IDS[:-1]])
    pixels = _pixels(model.config, dtype=dtype)
    forced = [31999, 31872, 31745]
    seen = []

    class KnownAction(LogitsProcessor):
        def __call__(self, input_ids, scores):
            seen.append(input_ids.clone())
            position = input_ids.shape[1] - len(TOKEN_IDS)
            assert 0 <= position < 3
            output = torch.full_like(scores, -torch.inf)
            output[:, forced[position]] = 0
            return output

    action = model.predict_action(ids, pixel_values=pixels, attention_mask=torch.ones_like(ids),
                                  do_sample=False, logits_processor=LogitsProcessorList([KnownAction()]))
    normalized = np.array([-254 / 255, 0, 254 / 255])
    expected = np.array([3 * (normalized[0] + 1) - 2, 3, normalized[2]])
    np.testing.assert_allclose(action, expected, atol=2e-15, rtol=0)
    assert seen[0].tolist() == [TOKEN_IDS]
    assert model.get_action_dim() == 3
    assert model.get_action_stats() == NORMS["robot"]["action"]


def test_openvla_action_real_greedy_oracle_and_validation_retry():
    model = _model(text_config={"vocab_size": 32064})
    ids = torch.tensor([TOKEN_IDS])
    pixels = _pixels(model.config)
    generated = model.generate(ids, pixel_values=pixels, max_new_tokens=3, do_sample=False)
    normalized = []
    for token in generated[0, -3:].tolist():
        index = max(0, min(254, 31999 - token))
        normalized.append(-1 + (index + 0.5) * 2 / 255)
    low, high = [-2, 1, 9], [4, 5, 13]
    expected = [low[i] + (normalized[i] + 1) * (high[i] - low[i]) / 2 if i < 2 else normalized[i]
                for i in range(3)]
    np.testing.assert_allclose(model.predict_action(ids, pixel_values=pixels, do_sample=False), expected, atol=2e-15)
    for kwargs, message in [({"unnorm_key": "missing"}, "Unknown unnorm_key"),
                            ({"input_ids": ids.expand(2, -1)}, "one nonempty"),
                            ({"input_ids": ids[:, :0]}, "one nonempty")]:
        with pytest.raises(ValueError, match=message):
            model.predict_action(**{"input_ids": ids, "pixel_values": pixels, **kwargs})
    model.norm_stats["other"] = copy.deepcopy(NORMS["robot"])
    with pytest.raises(ValueError, match="Choose an unnorm_key"):
        model.predict_action(ids, pixel_values=pixels)
    model.norm_stats.pop("other")
    model.norm_stats["robot"]["action"]["q99"] = [4, 5]
    with pytest.raises(ValueError, match="incompatible shapes"):
        model.predict_action(ids, pixel_values=pixels)
    model.norm_stats["robot"]["action"]["q99"] = [4, 5, 13]
    np.testing.assert_allclose(model.predict_action(ids, pixel_values=pixels, do_sample=False), expected, atol=2e-15)


@pytest.mark.parametrize("llm,offload,fused", [
    ("llama2-7b-pure", False, True), ("mistral-v0.1-7b-pure", True, False), ("phi-2-3b", False, False),
])
def test_openvla_public_image_awq_pack_and_two_exact_reloads(tmp_path, monkeypatch, llm, offload, fused):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    source = tmp_path / "source"
    source.mkdir()
    phi = llm == "phi-2-3b"
    original = _model(llm=llm, fused=fused, dtype=torch.float16,
                      text_config={"vocab_size": 32064, "qk_layernorm": phi})
    original.save_pretrained(source)
    _write_assets(source, original.config)
    linears = {"self_attn." + name + "_proj" for name in ("q", "k", "v")}
    linears |= {"self_attn.dense", "mlp.fc1", "mlp.fc2"} if phi else {
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    expected_paths = {f"language_model.model.layers.{i}.{name}" for i in range(2) for name in linears}
    # AWQ folds its activation scales into the Llama/Mistral pre-norm gains.
    folded = set() if phi else {f"language_model.model.layers.{i}.{name}.weight" for i in range(2)
                                for name in ("input_layernorm", "post_attention_layernorm")}
    dense = {k: v.clone() for k, v in original.state_dict().items()
             if k.rsplit(".", 1)[0] not in expected_paths and k not in folded}
    config = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload,
                       offload_to_disk_path=str(tmp_path / "offload"))
    wrapper = GPTQModel.load(str(source), quantize_config=config, backend=BACKEND.TORCH, dtype=torch.float16,
                            trust_remote_code=False, attn_implementation="eager")
    image = Image.fromarray(np.arange(23 * 31 * 3, dtype=np.uint8).reshape(23, 31, 3))
    calibration = [[{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": "pick up the cup"},
    ]}] for img in (image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))]
    prepared = wrapper.prepare_dataset(calibration, batch_size=1)
    expected_capture = []
    with torch.inference_mode():
        for example in prepared:
            patches = _image_equations(original, example["pixel_values"].half()).half()
            text = original.get_input_embeddings()(example["input_ids"])
            expected_capture.append(torch.cat((text[:, :1], patches, text[:, 1:]), 1))
    captures = []
    capture_inputs = StageInputsCapture.cache_inputs

    def observe_capture(self, *args, **kwargs):
        cache = capture_inputs(self, *args, **kwargs)
        captures.extend(values[0].cpu().clone() for values in cache.layer_inputs)
        return cache

    monkeypatch.setattr(StageInputsCapture, "cache_inputs", observe_capture)
    awq_layer = wrapper.model.language_model.model.layers[0]
    awq_features = {name: torch.zeros(1, 2, module.in_features) for name, module in awq_layer.named_modules()
                    if name in linears}
    awq_sources = [get_op_name(awq_layer, node["prev_op"])
                   for node in wrapper.awq_get_modules_for_scaling(awq_layer, awq_features, {})]
    # Phi's parallel norm and GELU leave only `v_proj -> dense`; GQA (two KV
    # heads) narrows it below `dense`'s input, so the processor drops it and
    # Phi takes the plain AWQ path.
    assert awq_sources == (["self_attn.v_proj"] if phi else [
        "input_layernorm", "self_attn.v_proj", "post_attention_layernorm", "mlp.up_proj",
    ])
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert len(captures) == 2
    for actual, expected in zip(captures, expected_capture):
        assert actual.shape == (1, len(TOKEN_IDS) + 4, 32)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.02)
        assert expected[:, 1:5].abs().max() > 1e-4
    assert {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected_paths
    for key, value in dense.items():
        assert torch.equal(wrapper.model.state_dict()[key], value), key
    sample = {k: v.half() if k == "pixel_values" else v for k, v in prepared[0].items()}
    with torch.inference_mode():
        logits = wrapper.model(**sample, use_cache=False).logits
    assert torch.isfinite(logits).all()
    packed_and_dense = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
    for cycle in range(2):
        saved = tmp_path / f"quantized-{cycle}"
        wrapper.save(str(saved))
        assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
        wrapper = GPTQModel.load(str(saved), device="cpu", backend=BACKEND.TORCH, dtype=torch.float16,
                                trust_remote_code=True, attn_implementation="eager")
        assert type(wrapper.model) is OpenVLAForActionPrediction
        assert set(wrapper.model.state_dict()) == set(packed_and_dense)
        for key, value in packed_and_dense.items():
            assert torch.equal(wrapper.model.state_dict()[key], value), key
        with torch.inference_mode():
            torch.testing.assert_close(wrapper.model(**sample, use_cache=False).logits, logits, atol=0, rtol=0)
        assert wrapper.processor(PROMPT, image)["input_ids"].tolist() == [TOKEN_IDS]
