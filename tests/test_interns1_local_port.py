# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real tiny Intern-S1 models: policy, multimodal numerics, cache and serialization."""

import io
import json

import pytest
import sentencepiece as spm
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GotOcr2ImageProcessor
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.interns1 import InternS1QModel
from gptqmodel.models.ports.interns1.configuration_interns1 import InternS1Config
from gptqmodel.models.ports.interns1.modeling_interns1 import InternS1ForConditionalGeneration
from gptqmodel.models.ports.interns1.processing_interns1 import InternS1Processor
from gptqmodel.models.ports.interns1.tokenization_interns1 import InternS1Tokenizer, bytes_to_unicode
from gptqmodel.models.ports.interns1.video_processing_interns1 import InternS1VideoProcessor
from gptqmodel.utils.hf import load_hf_tokenizer, load_model_config, resolve_trust_remote_code
from gptqmodel.utils.security import BYPASS_ENV


def tiny_config(moe=False, **kwargs):
    return InternS1Config(
        text_config={"model_type": "qwen3_moe" if moe else "qwen3", "vocab_size": 512,
                         "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "num_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
                         "pad_token_id": 0, "eos_token_id": 2, "bos_token_id": 1},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                           "num_attention_heads": 4, "image_size": 8, "patch_size": 2,
                           "norm_type": "rms_norm" if moe else "layer_norm", "use_qk_norm": moe,
                           "attention_bias": not moe, "drop_path_rate": .1},
        image_token_id=4, image_seq_length=4, **kwargs,
    )


def hostile_metadata(path, config):
    config.save_pretrained(path)
    payload = json.loads((path / "config.json").read_text())
    mapping = {"AutoConfig": "evil.Config", "AutoModelForCausalLM": "evil.Model"}
    payload["auto_map"] = mapping
    payload["text_config"]["auto_map"] = mapping
    payload["vision_config"]["auto_map"] = mapping
    (path / "config.json").write_text(json.dumps(payload))
    (path / "evil.py").write_text("raise AssertionError('checkpoint Python executed')\n")


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_selected_configuration_never_executes_checkpoint_code(tmp_path, monkeypatch, trust, bypass, moe):
    original = InternS1ForConditionalGeneration(tiny_config(moe)).eval()
    original.save_pretrained(tmp_path, safe_serialization=True)
    hostile_metadata(tmp_path, tiny_config(moe))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")  # Existing security bypass, not a new behavior switch.
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    selected = load_model_config(tmp_path, trust_remote_code=trust)
    assert type(selected) is InternS1Config
    assert resolve_trust_remote_code(tmp_path, trust_remote_code=trust) is False
    with torch.device("meta"):
        model = InternS1QModel.loader.from_config(selected, trust_remote_code=trust)
    assert type(model) is InternS1ForConditionalGeneration
    assert model.model.language_model.layers[0].self_attn.q_proj.weight.device.type == "meta"
    loaded = InternS1QModel.loader.from_pretrained(tmp_path, trust_remote_code=trust, code_revision="never").eval()
    assert type(loaded) is InternS1ForConditionalGeneration
    ids = torch.tensor([[1, 17, 18, 19]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(ids).logits, original(ids).logits, rtol=0, atol=0)


def test_unknown_nested_architecture_fails_then_same_path_retries(tmp_path):
    payload = tiny_config().to_dict()
    payload["text_config"]["model_type"] = "untrusted_architecture"
    (tmp_path / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires qwen3"):
        load_model_config(tmp_path)
    hostile_metadata(tmp_path, tiny_config())
    assert type(load_model_config(tmp_path)) is InternS1Config


def native_text_reference(model):
    config = model.config.text_config
    reference = (Qwen3MoeModel if model.is_moe_model else Qwen3Model)._from_config(
        config, dtype=next(model.parameters()).dtype,
    ).eval()
    weights = model.model.language_model.state_dict()
    if model.is_moe_model:
        weights = dict(weights)
        for index in range(config.num_hidden_layers):
            if not hasattr(model.model.language_model.layers[index].mlp, "experts"):
                continue
            prefix = f"layers.{index}.mlp.experts."
            weights[prefix + "gate_up_proj"] = torch.stack([
                torch.cat([weights.pop(f"{prefix}{expert}.gate_proj.weight"),
                           weights.pop(f"{prefix}{expert}.up_proj.weight")])
                for expert in range(config.num_experts)
            ])
            weights[prefix + "down_proj"] = torch.stack([
                weights.pop(f"{prefix}{expert}.down_proj.weight") for expert in range(config.num_experts)
            ])
    reference.load_state_dict(weights, strict=True)
    return reference


@pytest.mark.parametrize("moe", [False, True, "mlp_only", "sparse_step", "all_dense"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_text_logits_cache_generation_and_safetensors(tmp_path, moe, dtype):
    torch.manual_seed(17)
    config = tiny_config(moe)
    if moe == "mlp_only":
        config.text_config.mlp_only_layers = [0]
    elif moe == "sparse_step":
        config.text_config.decoder_sparse_step = 2
    elif moe == "all_dense":
        config.text_config.mlp_only_layers = [0, 1]
    model = InternS1ForConditionalGeneration._from_config(config, dtype=dtype).eval()
    reference = native_text_reference(model)
    ids = torch.tensor([[0, 11, 12, 13, 14], [21, 22, 23, 24, 25]])
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    tolerance = 2e-3 if dtype != torch.float32 else 2e-6
    with torch.inference_mode():
        output = model(ids, attention_mask=mask, use_cache=True, output_router_logits=moe)
        expected = model.lm_head(reference(ids, attention_mask=mask, use_cache=False).last_hidden_state)
        torch.testing.assert_close(output.logits, expected, atol=tolerance, rtol=tolerance)
        if moe:
            assert len(output.router_logits or ()) == sum(
                hasattr(layer.mlp, "experts") for layer in model.model.language_model.layers
            )
        cached = model(ids[:, -1:], attention_mask=torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), 1),
                       past_key_values=output.past_key_values, use_cache=True)
        whole = model(torch.cat((ids, ids[:, -1:]), 1),
                      attention_mask=torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), 1), use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1], whole.logits[:, -1], atol=tolerance, rtol=tolerance)
        assert cached.past_key_values.get_seq_length() == 6
        generated = model.generate(ids[1:], max_new_tokens=3, do_sample=False)
        uncached = model.generate(ids[1:], max_new_tokens=3, do_sample=False, use_cache=False)
        torch.testing.assert_close(generated, uncached)
        static = model.generate(ids[1:], max_new_tokens=3, do_sample=False,
                                cache_implementation="static", disable_compile=True)
        torch.testing.assert_close(generated, static)
        model.save_pretrained(tmp_path, safe_serialization=True)
        assert list(tmp_path.glob("*.safetensors")) and not list(tmp_path.glob("*.bin"))
        restored = InternS1QModel.loader.from_pretrained(tmp_path, dtype=dtype).eval()
        assert restored.state_dict().keys() == model.state_dict().keys()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored(ids, attention_mask=mask).logits, output.logits, rtol=0, atol=0)
        restored.save_pretrained(tmp_path / "again", safe_serialization=True)
        again = InternS1QModel.loader.from_pretrained(tmp_path / "again", dtype=dtype).eval()
        torch.testing.assert_close(again(ids, attention_mask=mask).logits, output.logits, rtol=0, atol=0)


def vision_reference(vision, pixels):
    weights = vision.state_dict()

    def linear(hidden, path):
        return F.linear(hidden, weights[path + ".weight"], weights.get(path + ".bias"))

    def norm(hidden, path):
        weight = weights.get(path + ".weight")
        if weight is None:
            return hidden
        if path + ".bias" in weights:
            return F.layer_norm(hidden, weight.shape, weight, weights[path + ".bias"], vision.config.layer_norm_eps)
        eps = 1e-6 if path.endswith(("q_norm", "k_norm")) else vision.config.layer_norm_eps
        return weight * hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)

    hidden = F.conv2d(pixels, weights["embeddings.patch_embeddings.projection.weight"],
                      weights["embeddings.patch_embeddings.projection.bias"], stride=vision.config.patch_size)
    hidden = hidden.flatten(2).transpose(1, 2)
    hidden = torch.cat([weights["embeddings.cls_token"].expand(pixels.shape[0], -1, -1), hidden], 1)
    hidden = hidden + weights["embeddings.position_embeddings"]
    heads = vision.config.num_attention_heads
    for index in range(vision.config.num_hidden_layers):
        path = f"encoder.layer.{index}"
        source = norm(hidden, path + ".layernorm_before")
        qkv = [norm(linear(source, path + f".attention.{role}_proj"), path + f".attention.{role}_norm")
               if role != "v" else linear(source, path + ".attention.v_proj") for role in ("q", "k", "v")]
        q, k, v = [value.reshape(*source.shape[:2], heads, -1).transpose(1, 2) for value in qkv]
        scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** .5
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape_as(source)
        hidden = hidden + weights[path + ".lambda_1"] * linear(attended, path + ".attention.projection_layer")
        source = norm(hidden, path + ".layernorm_after")
        source = linear(F.gelu(linear(source, path + ".mlp.fc1")), path + ".mlp.fc2")
        hidden = hidden + weights[path + ".lambda_2"] * source
    return hidden


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_image_features_match_manual_patch_shuffle_and_native_text(moe, attention):
    torch.manual_seed(41)
    model = InternS1ForConditionalGeneration._from_config(tiny_config(moe), attn_implementation=attention).eval()
    ids = torch.tensor([[1, 4, 4, 4, 4, 17], [1, 4, 4, 4, 4, 18]])
    pixels = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        vision = vision_reference(model.vision_tower, pixels)
        torch.testing.assert_close(model.vision_tower(pixels).last_hidden_state, vision, rtol=2e-6, atol=2e-6)
        patches = vision[:, 1:].reshape(2, 4, 4, 16)
        # Independent spatial unshuffle: each output channel contains one 2x2 block.
        grouped = torch.stack([
            torch.cat([patches[:, row + dy, col + dx] for dy in range(2) for dx in range(2)], -1)
            for row in range(0, 4, 2) for col in range(0, 4, 2)
        ], 1)
        projected = model.multi_modal_projector(grouped)
        torch.testing.assert_close(model.get_image_features(pixels), projected, rtol=2e-6, atol=2e-6)
        embedded = model.get_input_embeddings()(ids)
        embedded[:, 1:5] = projected
        expected = model.lm_head(native_text_reference(model)(inputs_embeds=embedded).last_hidden_state)
        output = model(ids, pixel_values=pixels, use_cache=True)
        torch.testing.assert_close(output.logits, expected, atol=2e-6, rtol=2e-6)
        with pytest.raises(ValueError, match="features and image tokens"):
            model(ids[:, 1:], pixel_values=pixels[:1])
        # Retry after malformed media; no shared mutable image/cache state survives the failure.
        torch.testing.assert_close(model(ids, pixel_values=pixels).logits, output.logits, rtol=0, atol=0)
        prefill_calls = []
        handle = model.vision_tower.register_forward_hook(lambda *args: prefill_calls.append(True))
        generated = model.generate(ids, pixel_values=pixels, max_new_tokens=3, do_sample=False, eos_token_id=None)
        assert len(prefill_calls) == 1
        prefill_calls.clear()
        uncached = model.generate(ids, pixel_values=pixels, max_new_tokens=3, do_sample=False, use_cache=False,
                                  eos_token_id=None)
        assert len(prefill_calls) == 3
        prefill_calls.clear()
        static = model.generate(ids, pixel_values=pixels, max_new_tokens=3, do_sample=False,
                                cache_implementation="static", disable_compile=True, eos_token_id=None)
        assert len(prefill_calls) == 1
        handle.remove()
        torch.testing.assert_close(generated, uncached)
        torch.testing.assert_close(generated, static)


@pytest.fixture
def tokenizer_assets(tmp_path):
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(["CCO carbon acid MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"] * 4),
                                 model_writer=buffer, vocab_size=32, hard_vocab_limit=False, num_threads=1,
                                 minloglevel=2, add_dummy_prefix=False)
    for name in ("SMILES", "IUPAC", "FASTA"):
        (tmp_path / f"tokenizer_{name}.model").write_bytes(buffer.getvalue())
    vocab = {value: index for index, value in enumerate(bytes_to_unicode().values())}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = InternS1Tokenizer(vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt"),
                                  extra_special_tokens={"context_image_token": "<IMG_CONTEXT>",
                                                        "start_image_token": "<img>", "end_image_token": "</img>",
                                                        "video_token": "<video>"})
    tokenizer.save_pretrained(tmp_path)
    GotOcr2ImageProcessor(size={"height": 8, "width": 8}, crop_to_patches=True, min_patches=1,
                         max_patches=2, image_mean=[.485, .456, .406], image_std=[.229, .224, .225]).save_pretrained(tmp_path)
    InternS1VideoProcessor(size={"height": 8, "width": 8}, do_sample_frames=False).save_pretrained(tmp_path)
    config = tiny_config()
    config.image_token_id = tokenizer.context_image_token_id
    config.save_pretrained(tmp_path)
    (tmp_path / "chat_template.jinja").write_text("{% for message in messages %}{% for c in message['content'] %}"
                                                  "{% if c['type'] == 'image' %}<IMG_CONTEXT>"
                                                  "{% elif c['type'] == 'video' %}<video>"
                                                  "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}")
    return tmp_path


def test_real_processor_media_failure_retry_and_domain_tokenizer(tokenizer_assets):
    processor = InternS1Processor.from_pretrained(tokenizer_assets, local_files_only=True)
    tokenizer = processor.tokenizer
    ids = tokenizer("<SMILES>CCO</SMILES>")["input_ids"]
    assert tokenizer.decode(ids) == "<SMILES>CCO</SMILES>"
    image = Image.new("RGB", (8, 8), color=(55, 77, 99))
    with pytest.raises((ValueError, IndexError)):
        processor(text="no image placeholder", images=image, return_tensors="pt")
    batch = processor(text="<IMG_CONTEXT> describe", images=image, return_tensors="pt")
    assert batch["pixel_values"].shape == (1, 3, 8, 8)
    assert (batch["input_ids"] == tokenizer.context_image_token_id).sum() == 4
    pixels = batch["pixel_values"]
    expected = (torch.tensor([55, 77, 99]) / 255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(pixels[0, :, 0, 0], expected)
    video = torch.stack([torch.full((3, 8, 8), i, dtype=torch.uint8) for i in (20, 40, 60)])
    video_batch = processor(text="<video> explain", videos=[video], return_tensors="pt")
    assert video_batch["pixel_values"].shape == (3, 3, 8, 8)
    assert (video_batch["input_ids"] == tokenizer.context_image_token_id).sum() == 12
    processor.save_pretrained(tokenizer_assets / "processor")
    tiny_config().save_pretrained(tokenizer_assets / "processor")
    again = InternS1Processor.from_pretrained(tokenizer_assets / "processor", local_files_only=True)
    actual = again(text="<IMG_CONTEXT> describe", images=image, return_tensors="pt")
    for key in batch:
        torch.testing.assert_close(actual[key], batch[key], rtol=0, atol=0)


def test_processor_multiple_variable_videos_keep_every_frame_and_interleaving(tokenizer_assets):
    processor = InternS1Processor.from_pretrained(tokenizer_assets)
    first = torch.full((3, 3, 8, 8), 40, dtype=torch.uint8)
    second = torch.full((2, 3, 8, 8), 170, dtype=torch.uint8)
    image = Image.new("RGB", (8, 8), color=(90, 90, 90))
    single = [processor(text="<video>", videos=[video])["pixel_values"] for video in (first, second)]
    middle = processor(text="<IMG_CONTEXT>", images=image)["pixel_values"]
    combined = processor(text="<video> then <IMG_CONTEXT> then <video>", images=image, videos=[first, second])
    expected = torch.cat([torch.stack([torch.as_tensor(frame) for frame in patches])
                          for patches in (single[0], middle, single[1])])
    torch.testing.assert_close(torch.as_tensor(combined["pixel_values"]), expected, rtol=0, atol=0)
    assert combined["input_ids"][0].count(processor.image_token_id) == 24


def reject_checkpoint_discovery(*args, **kwargs):
    raise AssertionError("package-owned Intern-S1 reentered checkpoint Auto discovery")


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("moe", [False, True])
def test_public_load_owned_scientific_tokenizer_processor_and_media(tokenizer_assets, monkeypatch, trust, bypass, moe):
    original_tokenizer = InternS1Tokenizer.from_pretrained(tokenizer_assets)
    config = tiny_config(moe)
    config.image_token_id = original_tokenizer.context_image_token_id
    original = InternS1ForConditionalGeneration._from_config(config).eval()
    original.save_pretrained(tokenizer_assets, safe_serialization=True)
    hostile_metadata(tokenizer_assets, config)
    for filename in ("tokenizer_config.json", "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        path = tokenizer_assets / filename
        metadata = json.loads(path.read_text()) if path.exists() else {}
        metadata["auto_map"] = {"AutoTokenizer": ["evil.InternS1Tokenizer", None], "AutoProcessor": "evil.Processor"}
        path.write_text(json.dumps(metadata))
    if bypass:
        monkeypatch.setenv(BYPASS_ENV, "1")
    else:
        monkeypatch.delenv(BYPASS_ENV, raising=False)
    for auto in (AutoConfig, AutoTokenizer, AutoProcessor):
        monkeypatch.setattr(auto, "from_pretrained", reject_checkpoint_discovery)
    loaded = GPTQModel.load(str(tokenizer_assets), device="cpu", dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded.model) is InternS1ForConditionalGeneration
    assert type(loaded.tokenizer.tokenizer) is InternS1Tokenizer
    assert type(loaded.processor) is InternS1Processor
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    for prompt in ("science café 中文", "<SMILES>CCO</SMILES>", "<IUPAC>carbon acid</IUPAC>",
                   "<FASTA_AUTO_DETECT>MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY</FASTA_AUTO_DETECT>"):
        ids = original_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        actual = loaded.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assert actual == ids
        assert loaded.tokenizer.decode(actual) == original_tokenizer.decode(ids)
    image = Image.new("RGB", (8, 8), color=(55, 77, 99))
    batch = loaded.processor(text="<IMG_CONTEXT> describe", images=image, return_tensors="pt")
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**batch).logits, original(**batch).logits, rtol=0, atol=0)
        generated = loaded.model.generate(**batch, max_new_tokens=2, do_sample=False)
        expected = original.generate(**batch, max_new_tokens=2, do_sample=False)
        torch.testing.assert_close(generated, expected)


@pytest.mark.parametrize("asset", InternS1Tokenizer.vocab_files_names.values())
def test_owned_tokenizer_missing_science_asset_fails_then_same_path_retries(tokenizer_assets, monkeypatch, asset):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", reject_checkpoint_discovery)
    config = load_model_config(tokenizer_assets)
    path = tokenizer_assets / asset
    content = path.read_bytes()
    path.unlink()
    with pytest.raises(RuntimeError, match=asset):
        load_hf_tokenizer(str(tokenizer_assets), model_config=config)
    path.write_bytes(content)
    tokenizer = load_hf_tokenizer(str(tokenizer_assets), model_config=config)
    assert tokenizer.decode(tokenizer("<IUPAC>carbon acid</IUPAC>")["input_ids"]) == "<IUPAC>carbon acid</IUPAC>"
