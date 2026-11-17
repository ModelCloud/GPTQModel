# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The flat Apple LLaVA-Qwen2 checkpoint must not load checkpoint Python."""

# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from gptqmodel.models.definitions.llava_qwen2 import LlavaQwen2QModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.llava_qwen2.configuration_llava_qwen2 import LlavaConfig
from gptqmodel.models.ports.llava_qwen2.modeling_llava_qwen2 import LlavaQwen2ForCausalLM
from gptqmodel.models.ports.llava_qwen2.processing_llava_qwen2 import LlavaQwen2Processor


def _config(tower="mobileclip", **kwargs):
    vision = {"layers": [1] * 5, "embed_dims": [16, 16, 32, 32, 32],
              "mlp_ratios": [1] * 5, "projection_dim": 8}
    values = {"vocab_size": 48, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 128,
                  "mm_vision_tower": "mobileclip_l_128", "mm_hidden_size": 64, "vision_config": vision,
                  "attn_implementation": "eager", "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
                  "tie_word_embeddings": False, "tokenizer_model_max_length": 128}
    if tower == "clip":
        values.update(mm_vision_tower="openai/clip-vit-base-patch16", mm_hidden_size=32,
                      vision_config={"image_size": 16, "patch_size": 8, "hidden_size": 32, "intermediate_size": 64,
                                         "num_hidden_layers": 2, "num_attention_heads": 4})
    values.update(kwargs)
    return LlavaConfig(**values)


def _model(tower="mobileclip", **kwargs):
    torch.manual_seed(412)
    model = LlavaQwen2ForCausalLM(_config(tower, **kwargs)).eval()
    # Nondegenerate scale/bias/conv values make the image path falsifiable.
    with torch.no_grad():
        for name, value in model.model.vision_tower.named_parameters():
            if "layer_scale" in name:
                value.fill_(0.3)
            elif value.ndim == 4:
                value.normal_(std=0.12)
                if "reparam" in name and name.endswith("weight"):
                    value.normal_(std=0.02)
                    for out in range(value.shape[0]):
                        value[out, out % value.shape[1], value.shape[2] // 2, value.shape[3] // 2] += 1.0
    return model


def _image(model, batch=1):
    size = model.get_vision_tower().image_processor.crop_size["height"]
    return torch.linspace(0, 1, batch * 3 * size * size).reshape(batch, 3, size, size)


def _write_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                   **{str(i): i for i in range(4, 48)}}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
                                       eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    tokenizer.save_pretrained(path)
    return tokenizer


def _vision_oracle(weights, images):
    """Functional pinned FastViTHD equations: no port modules or helpers."""
    prefix = "model.vision_tower.vision_tower.model."
    w = {key.removeprefix(prefix): value.double() for key, value in weights.items() if key.startswith(prefix)}

    def conv(x, name, stride=1, groups=1):
        kernel = w[name + ".weight"]
        return F.conv2d(x, kernel, w.get(name + ".bias"), stride=stride,
                        padding=kernel.shape[-1] // 2, groups=groups)

    def ffn(x, name):
        y = conv(x, name + ".conv.conv", groups=x.shape[1])
        p = name + ".conv.bn."
        y = (y - w[p + "running_mean"][None, :, None, None]) / torch.sqrt(
            w[p + "running_var"][None, :, None, None] + 1e-5)
        y = y * w[p + "weight"][None, :, None, None] + w[p + "bias"][None, :, None, None]
        return conv(F.gelu(conv(y, name + ".fc1")), name + ".fc2")

    x = images.double()
    for index, stride in enumerate((2, 2, 1)):
        x = F.gelu(conv(x, f"patch_embed.{index}.reparam_conv", stride, x.shape[1] if index == 1 else 1))
    for index in range(11):
        p = f"network.{index}"
        if index in (1, 3, 5, 8):
            x = F.gelu(conv(x, p + ".proj.0.lkb_reparam", 2, x.shape[1]))
            x = F.gelu(conv(x, p + ".proj.1.reparam_conv"))
        elif index in (6, 9):
            x = conv(x, p + ".reparam_conv", groups=x.shape[1])
        elif index in (0, 2, 4):
            p += ".0"
            x = conv(x, p + ".token_mixer.reparam_conv", groups=x.shape[1])
            x = x + w[p + ".layer_scale"] * ffn(x, p + ".convffn")
        else:
            p += ".0"
            y = (x - x.mean(1, keepdim=True)) * torch.rsqrt(x.var(1, correction=0, keepdim=True) + 1e-5)
            y = y * w[p + ".norm.weight"][:, None, None] + w[p + ".norm.bias"][:, None, None]
            batch, channels, height, width = y.shape
            y = y.flatten(2).transpose(1, 2)
            packed = F.linear(y, w[p + ".token_mixer.qkv.weight"])
            q, k, v = packed.reshape(batch, -1, 3, channels // 32, 32).permute(2, 0, 3, 1, 4)
            probs = (torch.einsum("bhnd,bhmd->bhnm", q, k) / 32 ** 0.5).softmax(-1)
            y = torch.einsum("bhnm,bhmd->bhnd", probs, v).transpose(1, 2).reshape(batch, -1, channels)
            y = F.linear(y, w[p + ".token_mixer.proj.weight"], w[p + ".token_mixer.proj.bias"])
            y = y.transpose(1, 2).reshape(batch, channels, height, width)
            x = x + w[p + ".layer_scale_1"] * y
            x = x + w[p + ".layer_scale_2"] * ffn(x, p + ".convffn")
    x = conv(x, "conv_exp.reparam_conv", groups=x.shape[1])
    scale = F.relu(conv(x.mean((-2, -1), keepdim=True), "conv_exp.se.reduce"))
    scale = conv(scale, "conv_exp.se.expand").sigmoid()
    return F.gelu(x * scale).flatten(2).transpose(1, 2)


def _decoder_oracle(weights, config, embeddings, mask, positions):
    """Independent float64 Qwen2 GQA, half-RoPE, RMSNorm, SwiGLU equations."""
    w = {key: value.double() for key, value in weights.items()}
    x = embeddings.double()
    batch, length, _ = x.shape
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    dim = config.hidden_size // heads

    def linear(x, name):
        return F.linear(x, w[name + ".weight"], w.get(name + ".bias"))

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps) * w[name + ".weight"]

    inv = torch.pow(10000.0, -torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    theta = positions.double()[..., None] * inv

    def rotate(x):
        first, second = x.chunk(2, -1)
        cosine, sine = theta.cos()[:, None], theta.sin()[:, None]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)

    allowed = torch.ones(length, length, dtype=torch.bool).tril()[None, None] & mask[:, None, None].bool()
    for index in range(config.num_hidden_layers):
        p = f"model.layers.{index}"
        y = norm(x, p + ".input_layernorm")
        q = linear(y, p + ".self_attn.q_proj").reshape(batch, length, heads, dim).transpose(1, 2)
        k, v = [linear(y, p + ".self_attn." + name).reshape(batch, length, kv_heads, dim).transpose(1, 2)
                for name in ("k_proj", "v_proj")]
        q, k = rotate(q), rotate(k).repeat_interleave(heads // kv_heads, 1)
        v = v.repeat_interleave(heads // kv_heads, 1)
        scores = ((q @ k.transpose(-1, -2)) / dim ** 0.5).masked_fill(~allowed, -torch.inf)
        # Pad queries are not compared; finite fill mirrors eager attention.
        scores = scores.nan_to_num(neginf=torch.finfo(scores.dtype).min)
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, -1)
        x = x + linear(y, p + ".self_attn.o_proj")
        y = norm(x, p + ".post_attention_layernorm")
        x = x + linear(F.silu(linear(y, p + ".mlp.gate_proj")) * linear(y, p + ".mlp.up_proj"), p + ".mlp.down_proj")
    return linear(norm(x, "model.norm"), "lm_head")


def test_definition_owns_the_multimodal_loader():
    assert isinstance(LlavaQwen2QModel.loader, LocalModelLoader)
    assert LlavaQwen2QModel.loader.processor_class is not None


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("projector", ["linear", "mlp1x_gelu", "mlp2x_gelu", "mlp3x_gelu", "identity"])
def test_image_forward_matches_independent_vision_projector_and_decoder_oracles(kv_heads, projector):
    model = _model(num_key_value_heads=kv_heads, hidden_size=64, num_attention_heads=4, mm_projector_type=projector)
    images = _image(model)
    ids = torch.tensor([[4, -200, 8, 11]])
    weights = model.state_dict()
    with torch.inference_mode():
        vision = _vision_oracle(weights, images)
        torch.testing.assert_close(model.get_vision_tower()(images).double(), vision, atol=1e-7, rtol=2e-4)
        projected = vision
        if projector == "linear":
            projected = F.linear(vision, weights["model.mm_projector.weight"].double(), weights["model.mm_projector.bias"].double())
        elif projector != "identity":
            for index in range(int(projector[3])):
                if index:
                    projected = F.gelu(projected)
                p = f"model.mm_projector.{index * 2}"
                projected = F.linear(projected, weights[p + ".weight"].double(), weights[p + ".bias"].double())
        embeddings = torch.cat((weights["model.embed_tokens.weight"][ids[:, :1]].double(), projected,
                                weights["model.embed_tokens.weight"][ids[:, 2:]].double()), 1)
        positions = torch.arange(embeddings.shape[1])[None]
        expected = _decoder_oracle(weights, model.config, embeddings, torch.ones_like(positions), positions)
        actual = model(ids, images=images, output_hidden_states=True)
        torch.testing.assert_close(actual.logits.double(), expected, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(actual.hidden_states[0].double(), embeddings, atol=1e-7, rtol=2e-4)
        other = model(ids, images=1 - images).logits
        assert (actual.logits[:, -1] - other[:, -1]).abs().max() > 1e-7


@pytest.mark.parametrize("tower", ["mobileclip", "clip"])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_image_cache_generation_and_safetensors_round_trip(tmp_path, tower, implementation, dtype):
    model = _model(tower, attn_implementation=implementation).to(dtype)
    ids = torch.tensor([[4, -200, 8, 11]])
    images = _image(model).to(dtype)
    tolerance = 1e-6 if dtype == torch.float32 else 0.006
    with torch.inference_mode():
        full = model(ids, images=images)
        prefix = model(ids[:, :3], images=images)
        old_length = prefix.past_key_values.get_seq_length()
        continued = model(ids[:, 3:], images=images, past_key_values=prefix.past_key_values)
        torch.testing.assert_close(continued.logits, full.logits[:, -1:], atol=tolerance, rtol=tolerance)
        assert continued.past_key_values.get_seq_length() == old_length + 1
        for beams in (1, 3):
            kwargs = {"max_new_tokens": 3, "num_beams": beams, "eos_token_id": None, "do_sample": False}
            cached = model.generate(ids[:, :3], images=images, **kwargs)
            replay = model.generate(ids[:, :3], images=images, use_cache=False, **kwargs)
            assert cached.shape == (1, 3)
            assert torch.equal(cached, replay)
        perturbed = ids.clone()
        perturbed[:, -1] = 12
        torch.testing.assert_close(model(perturbed, images=images).logits[:, :-1], full.logits[:, :-1],
                                   atol=tolerance, rtol=tolerance)
        model.save_pretrained(tmp_path)
        assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
        reloaded = LlavaQwen2QModel.loader.from_pretrained(tmp_path, dtype=dtype, attn_implementation=implementation)
        assert set(reloaded.state_dict()) == set(model.state_dict())
        for key, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(reloaded(ids, images=images).logits, full.logits, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("merge", ["flat", "spatial", "spatial_unpad"])
def test_multicrop_multiimage_labels_positions_and_padding_match_explicit_assembly(padding_side, merge):
    model = _model("clip", mm_patch_merge_type=merge, tokenizer_padding_side=padding_side,
                   image_aspect_ratio="anyres", image_grid_pinpoints=[[32, 16]])
    ids = torch.tensor([[4, -200, 8, -200, 9], [4, -200, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    images = [_image(model, 3), 1 - _image(model, 3), _image(model)]
    sizes = [(32, 16), (32, 16), (16, 16)]
    with torch.inference_mode():
        _, positions, new_mask, _, embeddings, labels = model.prepare_inputs_labels_for_multimodal(
            ids, None, mask, None, ids, images, sizes)
        expected_features = []
        for crops in images:
            features = model.encode_images(crops)
            if merge == "flat":
                merged = features.flatten(0, 1)
            elif crops.shape[0] == 1:
                merged = features[0]
                if merge == "spatial_unpad":
                    merged = torch.cat((merged, model.model.image_newline[None]))
            else:
                # A 1x2 crop grid of 2x2 patch embeddings: row-major, not crop-major.
                rows = torch.cat((features[1].reshape(2, 2, -1), features[2].reshape(2, 2, -1)), 1)
                if merge == "spatial_unpad":
                    rows = torch.cat((rows, model.model.image_newline[None, None].expand(2, 1, -1)), 1)
                merged = torch.cat((features[0], rows.flatten(0, 1)))
            expected_features.append(merged)
        e = model.model.embed_tokens.weight
        expected_rows = [torch.cat((e[4:5], expected_features[0], e[8:9], expected_features[1], e[9:10])),
                         torch.cat((e[4:5], expected_features[2], e[9:10]))]
        for row, expected in enumerate(expected_rows):
            keep = new_mask[row].bool()
            torch.testing.assert_close(embeddings[row, keep], expected)
            assert torch.equal(positions[row, keep], torch.arange(expected.shape[0]))
            assert torch.equal(labels[row, keep][labels[row, keep] != -100], ids[row, mask[row].bool()][ids[row, mask[row].bool()] != -200])
            if padding_side == "left":
                assert new_mask[row, -1]
            else:
                assert new_mask[row, 0]
        actual = model(ids, images=images, image_sizes=sizes, labels=ids, attention_mask=mask)
        assert torch.isfinite(actual.loss)


def test_bad_images_fail_before_cache_and_allow_corrected_retry():
    model = _model()
    ids, images = torch.tensor([[4, -200, 8]]), _image(model)
    with pytest.raises(ValueError, match="Image placeholders"):
        model(ids)
    with pytest.raises(ValueError, match="expected 1, got 2"):
        model(ids, images=images.repeat(2, 1, 1, 1))
    with pytest.raises(ValueError, match="attention_mask"):
        model(ids, images=images, attention_mask=torch.ones(1, 2))
    with torch.inference_mode():
        actual = model(ids, images=images)
        assert actual.past_key_values.get_seq_length() == 6
        assert torch.isfinite(actual.logits).all()


@pytest.mark.parametrize("feature", ["patch", "cls_patch"])
def test_clip_selected_layer_matches_native_feature_tensor_and_projection(feature):
    model = _model("clip", mm_vision_select_feature=feature, mm_vision_select_layer=-2)
    images = _image(model)
    with torch.inference_mode():
        raw = model.get_vision_tower().vision_tower(images, output_hidden_states=True).hidden_states[-2]
        expected = raw[:, 1:] if feature == "patch" else raw
        torch.testing.assert_close(model.get_vision_tower()(images), expected)
        torch.testing.assert_close(model.encode_images(images), model.model.mm_projector(expected))
        assert expected.shape[1] == (4 if feature == "patch" else 5)


def test_s2_clip_matches_independent_chessboard_resample_oracle_and_checkpoint_owners():
    model = _model("clip", s2=True, s2_scales="16,32,48", mm_hidden_size=96)
    tower = model.get_vision_tower()
    images = _image(model)
    outputs = []
    with torch.inference_mode():
        for scale in (16, 32, 48):
            resized = F.interpolate(images.float(), size=scale, mode="bicubic")
            n = scale // 16
            # Separate slice-based oracle, without port split/merge methods.
            crops = torch.cat([resized[:, :, y:y + 16, x:x + 16] for y in range(0, scale, 16)
                               for x in range(0, scale, 16)])
            features = tower.vision_tower(crops, output_hidden_states=True).hidden_states[-2][:, 1:]
            rows = []
            for y in range(n):
                rows.append(torch.cat([features[y * n + x].reshape(2, 2, 32) for x in range(n)], 1))
            grid = torch.cat(rows, 0).permute(2, 0, 1)[None]
            outputs.append(F.interpolate(grid, size=2, mode="area"))
        expected = torch.cat(outputs, 1).flatten(2).transpose(1, 2)
        torch.testing.assert_close(tower(images), expected)
        assert model(torch.tensor([[4, -200, 8]]), images=images).logits.shape == (1, 6, 48)
    ordinary = _model("clip")
    assert set(tower.state_dict()) == set(ordinary.get_vision_tower().state_dict())


def test_vision_tower_config_alias_keeps_flat_mobileclip_parameter_paths():
    config = _config()
    values = config.to_dict()
    values["vision_tower"] = values.pop("mm_vision_tower")
    with torch.device("meta"):
        model = LlavaQwen2ForCausalLM(LlavaConfig(**values))
    assert model.config.mm_vision_tower == "mobileclip_l_128"
    assert "model.vision_tower.vision_tower.model.head.proj" in model.state_dict()
    assert _config(vision_tower="unknown/decoy").mm_vision_tower == "mobileclip_l_128"


@pytest.mark.parametrize("kwargs, message", [
    ({"mm_hidden_size": 4}, "mm_hidden_size"),
    ({"mm_projector_type": "mlp0x_gelu"}, "Unknown projector"),
    ({"mm_projector_type": "identity"}, "equal vision/text"),
    ({"mm_vision_tower": "mobileclip_unknown_128"}, "Unsupported MobileCLIP"),
    ({"mm_vision_tower": "untrusted/config", "vision_config": None}, "explicit vision_config"),
    ({"mm_patch_merge_type": "unknown"}, "mm_patch_merge_type"),
    ({"tokenizer_padding_side": "unknown"}, "tokenizer_padding_side"),
])
def test_invalid_architectures_fail_closed_without_remote_discovery(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model(**kwargs)


def test_text_only_forward_agrees_with_native_qwen2_and_dummy_image_row_consumption():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = _model("clip")
    native = Qwen2ForCausalLM(Qwen2Config(**model.config.to_dict())).eval()
    native.load_state_dict({k: v for k, v in model.state_dict().items()
                            if not k.startswith(("model.vision_tower.", "model.mm_projector."))}, strict=True)
    ids = torch.tensor([[4, 8, 11]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, native(ids).logits)
        with_dummy = model(ids, images=_image(model)).logits
        torch.testing.assert_close(with_dummy, native(ids).logits)
        assert torch.equal(model.generate(ids, max_new_tokens=3), native.generate(ids, max_new_tokens=3))


def test_left_padded_mixed_batch_image_and_text_generation_matches_separate_requests():
    model = _model("clip", tokenizer_padding_side="left")
    ids = torch.tensor([[4, -200, 8, 11], [0, 0, 4, 8]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    images = _image(model, 2)
    with torch.inference_mode():
        outputs = model(ids, images=images, attention_mask=mask)
        image_row = model(ids[:1], images=images[:1])
        text_row = model(ids[1:, 2:])
        torch.testing.assert_close(outputs.logits[0], image_row.logits[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(outputs.logits[1, -2:], text_row.logits[0], atol=1e-6, rtol=1e-5)
        for beams in (1, 3):
            kwargs = {"num_beams": beams, "do_sample": False, "max_new_tokens": 3, "eos_token_id": None}
            generated = model.generate(ids, images=images, attention_mask=mask, **kwargs)
            one = model.generate(ids[:1], images=images[:1], **kwargs)
            two = model.generate(ids[1:, 2:], images=images[1:], **kwargs)
            assert torch.equal(generated, torch.cat((one, two)))


def test_input_capture_preserves_image_sentinel_but_sanitizes_real_invalid_token_ids():
    model = _model("clip")
    wrapper = LlavaQwen2QModel.__new__(LlavaQwen2QModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["tokenizer"] = None
    ids = torch.tensor([[4, -200, -201, 48, 8]])
    assert wrapper._sanitize_input_ids_for_embeddings(ids).tolist() == [[4, -200, 0, 0, 8]]
    assert ids.tolist() == [[4, -200, -201, 48, 8]]
    assert LlavaQwen2QModel.get_modules_with_direct_meta_tensors(model) == ["model"]


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_config_model_tokenizer_and_processor_ignore_hostile_auto_maps(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel
    from gptqmodel.models.auto import check_and_get_model_definition
    from gptqmodel.utils.hf import load_model_config
    model = _model("clip")
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n")
    mapping = dict.fromkeys(("AutoConfig", "AutoModelForCausalLM", "AutoProcessor", "AutoImageProcessor"), "hostile.RemoteClass")
    mapping["AutoTokenizer"] = [None, "hostile.RemoteTokenizer"]
    for name in ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
        path = tmp_path / name
        payload = json.loads(path.read_text()) if path.exists() else {}
        payload["auto_map"] = mapping
        payload["processor_class"] = "RemoteProcessor"
        path.write_text(json.dumps(payload))
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    assert check_and_get_model_definition(str(tmp_path), trust_remote_code=trust) is LlavaQwen2QModel
    assert type(load_model_config(tmp_path, trust_remote_code=trust)) is LlavaConfig
    loaded = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded) is LlavaQwen2QModel
    assert type(loaded.model) is LlavaQwen2ForCausalLM
    assert type(loaded.processor) is LlavaQwen2Processor
    assert loaded.trust_remote_code is False
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    assert not marker.exists()
    with torch.inference_mode():
        ids = torch.tensor([[4, -200, 8]])
        images = _image(model)
        torch.testing.assert_close(loaded.model(ids, images=images).logits, model(ids, images=images).logits)
    assert not marker.exists()


@pytest.mark.parametrize("aspect", ["pad", "square", "anyres"])
def test_owned_processor_image_chat_ids_pixels_and_save_reload(tmp_path, aspect):
    model = _model("clip", image_aspect_ratio=aspect, image_grid_pinpoints=[[32, 16]])
    model.save_pretrained(tmp_path)
    tokenizer = _write_tokenizer(tmp_path)
    processor = LlavaQwen2QModel.loader.load_processor(tmp_path, config=model.config, tokenizer=tokenizer)
    image = Image.new("RGB", (30, 14), (50, 100, 150))
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "4 8"}]}]
    rendered = processor.apply_chat_template(conversation, add_generation_prompt=True)
    assert rendered == "user: <image>\n4 8\nassistant: "
    encoded = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True)
    assert encoded.input_ids.tolist() == [[3, 3, -200, 4, 8, 3, 3]]
    assert encoded.image_sizes == [(30, 14)]
    pixels = encoded.images[0]
    assert pixels.shape == ((3, 3, 16, 16) if aspect == "anyres" else (3, 16, 16))
    assert torch.isfinite(pixels).all()
    with torch.inference_mode():
        assert model(**encoded).logits.shape[-1] == 48
    processor.save_pretrained(tmp_path)
    reloaded = LlavaQwen2QModel.loader.load_processor(tmp_path)
    repeated = reloaded.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True)
    assert torch.equal(repeated.input_ids, encoded.input_ids)
    torch.testing.assert_close(repeated.images, encoded.images)
    assert not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("tower", ["mobileclip", "clip"])
@pytest.mark.parametrize("merge", ["flat", "spatial_unpad"])
def test_public_plain_awq_image_calibration_pack_infer_and_safetensors_reload(tmp_path, tower, merge):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
    size = 128 if tower == "mobileclip" else 16
    model = _model(tower, mm_patch_merge_type=merge,
                   image_aspect_ratio="square" if merge == "flat" else "anyres",
                   image_grid_pinpoints=[[size, size]])
    model.save_pretrained(tmp_path)
    _write_tokenizer(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                             backend=BACKEND.TORCH, dtype=torch.float16)
    size = model.get_vision_tower().image_processor.crop_size["height"]
    conversation = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (size, size), (50, 100, 150))},
                                                 {"type": "text", "text": "4 8 11"}]}]
    batch = wrapper.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True)
    assert wrapper.awq_get_modules_for_scaling(wrapper.model.model.layers[0], {}, {}) == []
    lengths = []
    handle = wrapper.model.model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append((kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).shape[1]),
        with_kwargs=True)
    try:
        wrapper.quantize(
            CalibrationConfig(dataset=[conversation, conversation], batch_size=1, min_length=1), backend=BACKEND.TORCH
        )
    finally:
        handle.remove()
    expected_length = batch.input_ids.shape[1] - 1 + (4 if merge == "flat" else 10)
    assert expected_length in lengths, "Calibration must expand -200 into actual image features"
    names = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    expected_names = {f"model.layers.{index}.{name}" for index in range(2) for name in
                      ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    assert names == expected_names
    with torch.inference_mode():
        expected = wrapper.model(**batch, use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert {name for name, module in reloaded.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected_names
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(**batch, use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
    assert set(reloaded.model.state_dict()) == set(wrapper.model.state_dict())
    for name, value in reloaded.model.state_dict().items():
        torch.testing.assert_close(value, wrapper.model.state_dict()[name], atol=0, rtol=0)
