# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ovis2.5 real CPU model, media equations, cache and public lifecycle regressions."""

import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import Qwen3ForCausalLM, SiglipImageProcessorPil

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.ovis2_5 import Ovis2_5QModel
from gptqmodel.models.ports.ovis2_5.configuration_ovis2_5 import Ovis2_5_Config
from gptqmodel.models.ports.ovis2_5.modeling_ovis2_5 import Ovis2_5
from gptqmodel.models.ports.ovis2_5.processing_ovis2_5 import Ovis2_5Processor


def _config(text=None, vision=None, **kwargs):
    llm = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
               "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "pad_token_id": 0,
               "bos_token_id": 1, "eos_token_id": 2, "max_position_embeddings": 128,
               "tie_word_embeddings": True, "attn_implementation": "eager"}
    vit = {"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2, "num_attention_heads": 2,
               "patch_size": 2, "image_size": 8, "hidden_stride": 2, "window_size": 8, "attn_implementation": "eager"}
    llm.update(text or {})
    vit.update(vision or {})
    kwargs.setdefault("min_pixels", 64)
    kwargs.setdefault("max_pixels", 64)
    return Ovis2_5_Config(llm_config=llm, vit_config=vit, visual_vocab_size=20, **kwargs)


def _model(**kwargs):
    torch.manual_seed(2508)
    return Ovis2_5(_config(**kwargs)).eval()


def _assets(path):
    # Tiny tokenizer is only a local-loader/calibration fixture. Real release tokenization is
    # separately checked against the pinned Qwen3 assets; it is not claimed by this fixture.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                   **{str(i): i for i in range(4, 64)}}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]",
                                        bos_token="[BOS]", unk_token="[UNK]")
    tokenizer.chat_template = ("{% for message in messages %}{% if message['content'] is string %}"
                               "{{ message['content'] }}{% else %}{% for item in message['content'] %}"
                               "{% if item['type'] == 'image' %}<image>{% elif item['type'] == 'video' %}<video>"
                               "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}")
    tokenizer.save_pretrained(path)
    SiglipImageProcessorPil(size={"height": 8, "width": 8}).save_pretrained(path)
    return tokenizer


def _inputs(dtype=torch.float32, video=False):
    grid = torch.tensor([[2 if video else 1, 4, 4]])
    count = int(grid.prod()) // 4
    ids = torch.tensor([[1, -303 if video else -301] + [-300] * count + [-304 if video else -302, 5, 6]])
    pixels = torch.linspace(-1.7, 1.3, int(grid.prod()) * 12, dtype=dtype).reshape(-1, 12)
    return {"input_ids": ids, "pixel_values": pixels, "grid_thws": grid, "attention_mask": torch.ones_like(ids)}


def _vision_oracle(model, pixels, grids):
    """Independent scalar-position NaViT oracle in canonical merge order, without window permutation."""
    c = model.config.vit_config
    weights = {k: v.double() for k, v in model.visual_tokenizer.state_dict().items()}
    prefix = "vit.vision_model."

    def linear(x, key):
        return F.linear(x, weights[prefix + key + ".weight"], weights.get(prefix + key + ".bias"))

    def norm(x, key):
        return F.layer_norm(x, (x.shape[-1],), weights[prefix + key + ".weight"],
                            weights[prefix + key + ".bias"], c.layer_norm_eps)

    patch = weights[prefix + "embeddings.patch_embedding.weight"].flatten(1)
    x = F.linear(pixels.double(), patch, weights[prefix + "embeddings.patch_embedding.bias"])
    coords, groups = [], []
    offset = 0
    for media, (t, h, w) in enumerate(grids.tolist()):
        positions = [(frame, bh + dh, bw + dw) for frame in range(t)
                     for bh in range(0, h, c.hidden_stride) for bw in range(0, w, c.hidden_stride)
                     for dh in range(c.hidden_stride) for dw in range(c.hidden_stride)]
        if c.preserve_original_pe:
            side = int(weights[prefix + "embeddings.position_embedding.weight"].shape[0] ** .5)
            pe = weights[prefix + "embeddings.position_embedding.weight"].reshape(side, side, -1).permute(2, 0, 1)[None]
            pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)[0]
            x[offset:offset + t * h * w] += torch.stack([pe[:, row, col] for _, row, col in positions])
        coords.extend((row, col) for _, row, col in positions)
        # Source windows are sized in merged groups, then expanded to patch coordinates.
        window = c.window_size // (c.patch_size * c.hidden_stride) * c.hidden_stride
        groups.extend((media, frame, row // window, col // window) for frame, row, col in positions)
        offset += t * h * w
    coords = torch.tensor(coords)
    d = c.hidden_size // c.num_attention_heads
    freq = coords[:, :, None].float() * (10000. ** (-torch.arange(0, d // 2, 2).float() / (d // 2)))
    freq = freq.flatten(1).repeat(1, 2)[:, None]
    full = c.fullatt_block_indexes
    if isinstance(full, str):
        full = [int(i) for i in full.split("|")]
    for i in range(c.num_hidden_layers):
        p = f"encoder.layers.{i}."
        y = norm(x, p + "layer_norm1")
        q, k, v = [linear(y, p + "self_attn." + key).reshape(-1, c.num_attention_heads, d)
                   for key in ("q_proj", "k_proj", "v_proj")]
        if c.use_rope:
            def rope(z):
                z = z.float()
                return (z * freq.cos() + torch.cat((-z[..., d // 2:], z[..., :d // 2]), -1) * freq.sin()).double()
            q, k = rope(q), rope(k)
        scores = torch.einsum("ihd,jhd->hij", q, k) / d ** .5
        labels = [entry[:2] if full is None or i in full else entry for entry in groups]
        allowed = torch.tensor([[a == b for b in labels] for a in labels])
        probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        y = torch.einsum("hij,jhd->ihd", probs, v).reshape_as(x)
        x = x + linear(y, p + "self_attn.out_proj")
        y = norm(x, p + "layer_norm2")
        x = x + linear(F.gelu(linear(y, p + "mlp.fc1"), approximate="tanh"), p + "mlp.fc2")
    # Source consumes the last intermediate encoder state, BEFORE post_layernorm.
    x = x.reshape(-1, c.hidden_size * c.hidden_stride ** 2)
    logits = F.linear(x, weights["head.0.weight"])
    logits = F.layer_norm(logits, (logits.shape[-1],), weights["head.1.weight"], weights["head.1.bias"], 1e-5)
    return F.pad(logits.softmax(-1), (0, 4)) @ model.vte.weight.double()


@pytest.mark.parametrize("linear,pe,rope,full", [(False, True, True, None), (False, True, True, "1"),
                                                (True, True, True, None), (True, False, False, [])])
@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("head_dim", [8, 64])
def test_custom_vision_and_soft_vocabulary_match_independent_equations(linear, pe, rope, full, video, head_dim):
    model = _model(vision={"num_patches": 16 if linear else -1, "preserve_original_pe": pe,
                          "use_rope": rope, "fullatt_block_indexes": full, "hidden_size": head_dim * 2,
                          "window_size": 4 if full is not None else 8})
    inputs = _inputs(video=video)
    expected = _vision_oracle(model, inputs["pixel_values"], inputs["grid_thws"])
    with torch.no_grad():
        actual = model.vte(model.visual_tokenizer(inputs["pixel_values"], inputs["grid_thws"]))
    torch.testing.assert_close(actual.double(), expected, atol=5e-7, rtol=3e-5)
    altered = inputs["pixel_values"].flip(0)
    with torch.no_grad():
        changed = model.vte(model.visual_tokenizer(altered, inputs["grid_thws"]))
    assert (changed - actual).abs().max() > 1e-5


@pytest.mark.parametrize("frames", [1, 2])
def test_partial_window_tiles_match_independent_equations(frames):
    # Window 14 / patch 2 / merge 2 rounds down to three merged groups: six patches.
    # A 4x8 grid has two nonempty windows, with padding along both grid axes.
    model = _model(vision={"hidden_size": 128, "window_size": 14, "fullatt_block_indexes": "1"})
    grids = torch.tensor([[frames, 4, 8]])
    pixels = torch.linspace(-1.3, 1.6, frames * 4 * 8 * 12).reshape(-1, 12)
    expected = _vision_oracle(model, pixels, grids)
    with torch.no_grad():
        actual = model.vte(model.visual_tokenizer(pixels, grids))
    torch.testing.assert_close(actual.double(), expected, atol=5e-7, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_forward_cache_generation_and_two_dense_safetensors_cycles(tmp_path, dtype, attention):
    model = _model(text={"attn_implementation": attention},
                   vision={"attn_implementation": attention, "hidden_size": 128}).to(dtype)
    inputs = _inputs(dtype=dtype, video=True)
    reference = Qwen3ForCausalLM(copy.deepcopy(model.config.llm_config)).to(dtype).eval()
    reference.load_state_dict(model.llm.state_dict())
    with torch.no_grad():
        embeds = model.get_wte()(inputs["input_ids"].clamp_min(0))
        embeds[inputs["input_ids"] == -300] = _vision_oracle(model, inputs["pixel_values"], inputs["grid_thws"]).to(dtype)
        embeds[inputs["input_ids"] == -303] = model.vte.weight[-2]
        embeds[inputs["input_ids"] == -304] = model.vte.weight[-1]
        expected = reference(inputs_embeds=embeds, use_cache=False).logits
        actual = model(**inputs, use_cache=True)
        tol = .002 if dtype == torch.float16 else .02 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual.logits, expected, atol=tol, rtol=tol)
        next_ids = torch.tensor([[7, 8, 9, 10]])
        continued = model(input_ids=next_ids, past_key_values=actual.past_key_values, use_cache=True).logits
        full_inputs = dict(inputs, input_ids=torch.cat([inputs["input_ids"], next_ids], -1))
        full_inputs.pop("attention_mask")
        torch.testing.assert_close(continued, model(**full_inputs, use_cache=False).logits[:, -4:], atol=tol, rtol=tol)
        for beams in (1, 3):
            got = model.generate(**inputs, num_beams=beams, max_new_tokens=3, do_sample=False)
            want = reference.generate(inputs_embeds=embeds, num_beams=beams, max_new_tokens=3, do_sample=False)
            assert torch.equal(got, want)
        tensors = {k: v.clone() for k, v in model.state_dict().items()}
        for iteration in range(2):
            path = tmp_path / str(iteration)
            model.save_pretrained(path)
            assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
            model = Ovis2_5.from_pretrained(path, dtype=dtype).eval()
            for key, value in tensors.items():
                assert torch.equal(model.state_dict()[key], value), key
            torch.testing.assert_close(model(**inputs, use_cache=False).logits, actual.logits, atol=tol, rtol=tol)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_hostile_maps_processor_and_repaired_checkpoint_retry(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import GPTQModel

    model = _model()
    model.save_pretrained(tmp_path)
    _assets(tmp_path)
    marker = tmp_path / "executed"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('remote code')\n")
    (tmp_path / "custom_generate").mkdir()
    (tmp_path / "custom_generate" / "generate.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint custom generation')\n")
    for filename in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        payload = json.loads((tmp_path / filename).read_text())
        payload["auto_map"] = {key: ["hostile.Bad", "hostile.Bad"] if key == "AutoTokenizer" else "hostile.Bad"
                               for key in ("AutoConfig", "AutoModelForCausalLM", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor")}
        if filename == "config.json":
            for name in ("llm_config", "vit_config"):
                payload[name]["auto_map"] = {"AutoConfig": "hostile.Bad", "AutoModel": "hostile.Bad"}
        (tmp_path / filename).write_text(json.dumps(payload))
    monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    if bypass is not None:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    checkpoint = tmp_path / "model.safetensors"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"broken safetensors")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|deserializ"):
        GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    checkpoint.write_bytes(original)
    loaded = GPTQModel.load(str(tmp_path), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded) is Ovis2_5QModel and type(loaded.model) is Ovis2_5
    assert type(loaded.processor) is Ovis2_5Processor
    assert loaded.model.text_tokenizer is loaded.tokenizer.tokenizer is loaded.processor.tokenizer
    inputs = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(**inputs).logits, model(**inputs).logits)
    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "4 5 6 7"}]}]
    ids, pixels, grid = loaded.model.preprocess_inputs(messages, min_pixels=64, max_pixels=64)
    assert ids.lt(0).sum() == 6 and pixels.shape == (16, 12) and grid.tolist() == [[1, 4, 4]]
    assert loaded.model.generate(inputs=ids, pixel_values=pixels, grid_thws=grid, max_new_tokens=2).shape == (1, 2)
    assert not marker.exists()


@pytest.mark.parametrize("assign", [False, True])
def test_direct_assignment_and_invalid_media_failure_then_retry(assign):
    reference = _model()
    inputs = _inputs()
    target = _model()
    with torch.no_grad():
        target.vte.weight.add_(1.)
    target.load_state_dict(reference.state_dict(), assign=assign)
    bad = dict(inputs, grid_thws=torch.tensor([[1, 2, 2]]))
    with pytest.raises(ValueError, match="atom count"):
        target(**bad)
    bad = dict(inputs, pixel_values=None)
    with pytest.raises(ValueError, match="placeholders require"):
        target(**bad)
    with torch.no_grad():
        torch.testing.assert_close(target(**inputs).logits, reference(**inputs).logits)


def test_patchification_pixels_and_video_frame_order(tmp_path):
    tokenizer = _assets(tmp_path)
    processor = Ovis2_5Processor.from_pretrained(tmp_path, config=_config(), tokenizer=tokenizer)
    frames = [Image.fromarray((np.arange(192).reshape(8, 8, 3) + i * 20).astype(np.uint8)) for i in range(3)]
    messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": "4 5"}]}]
    ids, pixels, grid = processor.preprocess_inputs(messages, min_pixels=64, max_pixels=64)
    expected = []
    for frame in frames:
        image = np.asarray(frame).astype(np.float32) / 127.5 - 1
        for row in (0, 4):
            for col in (0, 4):
                for dy in (0, 2):
                    for dx in (0, 2):
                        expected.append(image[row + dy:row + dy + 2, col + dx:col + dx + 2].transpose(2, 0, 1).reshape(-1))
    torch.testing.assert_close(pixels, torch.from_numpy(np.stack(expected)), atol=2e-7, rtol=1e-6)
    assert grid.tolist() == [[3, 4, 4]] and ids[0].tolist() == [-303] + [-300] * 12 + [-304, 4, 5]
    saved = tmp_path / "processor"
    processor.save_pretrained(saved)
    reloaded = Ovis2_5Processor.from_pretrained(saved)
    again = reloaded.preprocess_inputs(messages, min_pixels=64, max_pixels=64)
    assert all(torch.equal(a, b) for a, b in zip((ids, pixels, grid), again))


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_ragged_images_left_padded_rows_and_beams_match_native_text_oracle(attention):
    model = _model(text={"attn_implementation": attention}, vision={"attn_implementation": attention,
                   "fullatt_block_indexes": [1]})
    grids = torch.tensor([[1, 4, 4], [1, 4, 8]])
    pixels = torch.linspace(-1.9, 1.7, 48 * 12).reshape(48, 12)
    ids = torch.tensor([[0] * 4 + [1, -301] + [-300] * 4 + [-302, 5, 6],
                        [1, -301] + [-300] * 8 + [-302, 5, 6]])
    mask = ids.ne(0)
    with torch.no_grad():
        embeddings = model.get_wte()(ids.clamp_min(0))
        embeddings[ids == -300] = _vision_oracle(model, pixels, grids).float()
        embeddings[ids == -301] = model.vte.weight[-4]
        embeddings[ids == -302] = model.vte.weight[-3]
        expected = model.llm(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits
        actual = model(input_ids=ids, attention_mask=mask, pixel_values=pixels, grid_thws=grids, use_cache=True)
        torch.testing.assert_close(actual.logits, expected, atol=2e-5, rtol=2e-5)
        for beams in (1, 3):
            actual_ids = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                                        grid_thws=grids, max_new_tokens=3, num_beams=beams)
            expected_ids = model.llm.generate(inputs_embeds=embeddings, attention_mask=mask,
                                              max_new_tokens=3, num_beams=beams)
            assert torch.equal(actual_ids, expected_ids)


def test_budgeted_thinking_matches_source_two_native_generation_stages(tmp_path):
    model = _model()
    model.tokenizer = _assets(tmp_path)
    inputs = _inputs()
    with torch.no_grad():
        embeddings = model.get_wte()(inputs["input_ids"].clamp_min(0))
        embeddings[inputs["input_ids"] == -300] = _vision_oracle(
            model, inputs["pixel_values"], inputs["grid_thws"]).float()
        embeddings[inputs["input_ids"] == -301] = model.vte.weight[-4]
        embeddings[inputs["input_ids"] == -302] = model.vte.weight[-3]
        first = model.llm.generate(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"],
                                    max_new_tokens=1, do_sample=False)
        forced = model.tokenizer("\n\nConsidering the limited time by the user, I have to give the solution "
                                 "based on the thinking directly now.\n</think>\n\n", return_tensors="pt",
                                 return_attention_mask=False).input_ids
        appended = torch.cat([first, forced], -1)
        new_embeddings = torch.cat([embeddings, model.get_wte()(appended)], -2)
        new_mask = torch.cat([inputs["input_ids"], appended], -1).ne(model.tokenizer.pad_token_id)
        last = model.llm.generate(inputs_embeds=new_embeddings, attention_mask=new_mask,
                                  max_new_tokens=64 - appended.shape[-1], do_sample=False)
        got = model.generate(**inputs, enable_thinking=True, enable_thinking_budget=True,
                              thinking_budget=1, max_new_tokens=64, do_sample=False)
        assert torch.equal(got, torch.cat([appended, last], -1))
    with pytest.raises(ValueError, match="one prompt and tensor output"):
        model.generate(**inputs, enable_thinking=True, enable_thinking_budget=True,
                        return_dict_in_generate=True, max_new_tokens=64)
    model.generation_config.return_dict_in_generate = True
    with pytest.raises(ValueError, match="one prompt and tensor output"):
        model.generate(**inputs, enable_thinking=True, enable_thinking_budget=True, max_new_tokens=64)


@pytest.mark.parametrize("media", [False, True])
def test_saved_generation_policy_reaches_native_decoder_and_two_reloads(tmp_path, media):
    from transformers import GenerationConfig

    from gptqmodel import GPTQModel

    model = _model()
    # Non-defaults mirror the released policy's sampling and EOS-list structure.
    policy = GenerationConfig(do_sample=True, top_k=20, top_p=.95, temperature=.6,
                              repetition_penalty=1.05, eos_token_id=[2, 63], pad_token_id=0,
                              bos_token_id=1, max_new_tokens=3, return_dict_in_generate=True,
                              output_scores=True, multimodal_max_length=8192)
    model.generation_config = copy.deepcopy(policy)
    inputs = _inputs() if media else {"input_ids": torch.tensor([[1, 5, 6]])}
    for cycle in range(3):
        reference = Qwen3ForCausalLM(copy.deepcopy(model.config.llm_config)).eval()
        reference.load_state_dict(model.llm.state_dict())
        with torch.no_grad():
            ids = inputs["input_ids"]
            embeddings = reference.get_input_embeddings()(ids.clamp_min(0))
            if media:
                embeddings[ids == -300] = _vision_oracle(model, inputs["pixel_values"], inputs["grid_thws"]).float()
                embeddings[ids == -301] = model.vte.weight[-4]
                embeddings[ids == -302] = model.vte.weight[-3]
            torch.manual_seed(3525)
            expected = reference.generate(inputs_embeds=embeddings, generation_config=copy.deepcopy(policy))
            torch.manual_seed(3525)
            actual = model.generate(**inputs)
            assert hasattr(actual, "sequences"), "Saved root generation policy did not reach the native decoder"
            assert torch.equal(actual.sequences, expected.sequences)
            for got, want in zip(actual.scores, expected.scores, strict=True):
                torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
            # An explicit policy supersedes the saved one; EOS is the actual greedy first token.
            eos = int(reference(inputs_embeds=embeddings, use_cache=False).logits[0, -1].argmax())
            override = GenerationConfig(do_sample=False, eos_token_id=eos, pad_token_id=0, max_new_tokens=4)
            assert model.generate(**inputs, generation_config=override).tolist() == [[eos]]
            assert model.generate(**inputs, generation_config=override, eos_token_id=None,
                                  max_new_tokens=2).shape == (1, 2)
        assert model.generation_config.to_dict() == policy.to_dict()
        if cycle < 2:
            directory = tmp_path / str(cycle)
            model.save_pretrained(directory)
            _assets(directory)
            model = (GPTQModel.load(str(directory), device="cpu", dtype=torch.float32).model
                     if cycle == 0 else Ovis2_5.from_pretrained(directory)).eval()


@pytest.mark.parametrize("kv,bias", [(4, False), (2, True), (1, False)])
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_awq_capture_all_linears_and_two_exact_roundtrips(tmp_path, kv, bias, offload, monkeypatch):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.looper import awq_processor
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig

    model = _model(text={"num_key_value_heads": kv, "attention_bias": bias}).half()
    model.save_pretrained(tmp_path)
    _assets(tmp_path)
    quantized_weights = {f"llm.model.layers.{i}.{name}.weight" for i in range(2) for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    dense = {k: v.clone() for k, v in model.state_dict().items() if k not in quantized_weights}
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(
        device="cpu", group_size=16, fallback=None, offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload")), backend=BACKEND.TORCH, dtype=torch.float16)
    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    calibration = [[{"role": "user", "content": [{"type": "image", "image": image},
                                                  {"type": "text", "text": "4 5 6 7 8 9 10 11"}]}] for _ in range(2)]
    prepared = wrapper.prepare_dataset(calibration, batch_size=1, calibration_data_min_length=1)[0]
    with torch.no_grad():
        ids = prepared["input_ids"]
        expected_capture = model.get_wte()(ids.clamp_min(0))
        expected_capture[ids == -300] = _vision_oracle(model, prepared["pixel_values"], prepared["grid_thws"]).half()
        expected_capture[ids == -301] = model.vte.weight[-4]
        expected_capture[ids == -302] = model.vte.weight[-3]
    captured = []

    def capture(module, args, kwargs):
        captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    applied = []
    original_apply_scale = awq_processor.apply_scale

    def record_apply_scale(module, scales_list, *args, **kwargs):
        applied.extend((source, tuple(targets)) for source, targets, _scales, _loss in scales_list)
        return original_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", record_apply_scale)
    handle = wrapper.model.llm.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        wrapper.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    finally:
        handle.remove()
    assert len(captured) >= 2
    for actual in captured[:2]:
        torch.testing.assert_close(actual, expected_capture, atol=.002, rtol=.002)
    assert sum(isinstance(m, AwqTorchLinear) for m in wrapper.model.modules()) == 14
    groups = [
        ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ("mlp.up_proj", ("mlp.down_proj",)),
    ]
    if kv == 4:
        groups.insert(1, ("self_attn.v_proj", ("self_attn.o_proj",)))
    # Every decoder layer searched and folded its scale groups.
    assert applied == 2 * groups
    # Folding divides the two decoder norm gains; every other dense tensor
    # (vision tower, visual vocabulary, embeddings, Q/K norms) is untouched.
    folded = {f"llm.model.layers.{i}.{norm}.weight" for i in range(2)
              for norm in ("input_layernorm", "post_attention_layernorm")}
    for key, value in dense.items():
        if key not in folded:
            assert torch.equal(wrapper.model.state_dict()[key], value), key
    inputs = _inputs(dtype=torch.float16, video=True)
    with torch.no_grad():
        expected = wrapper.model(**inputs, use_cache=False).logits
        generated = wrapper.generate(**inputs, max_new_tokens=2, do_sample=False)
    tensors = {k: v.clone() for k, v in wrapper.model.state_dict().items()}
    for iteration in range(2):
        destination = tmp_path / f"quantized-{iteration}"
        wrapper.save_quantized(str(destination))
        wrapper = GPTQModel.load(str(destination), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        assert sum(isinstance(m, AwqTorchLinear) for m in wrapper.model.modules()) == 14
        assert wrapper.model.text_tokenizer is wrapper.processor.tokenizer
        for key, value in tensors.items():
            assert torch.equal(wrapper.model.state_dict()[key], value), key
        with torch.no_grad():
            torch.testing.assert_close(wrapper.model(**inputs, use_cache=False).logits, expected, atol=2e-4, rtol=.002)
            assert torch.equal(wrapper.generate(**inputs, max_new_tokens=2, do_sample=False), generated)
