# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real CPU qualification for the package-owned multimodal Agnes port."""

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.agnes import AgnesQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.agnes.configuration_agnes import AgnesConfig, AgnesTextConfig, AgnesVisionConfig
from gptqmodel.models.ports.agnes.image_processing_agnes import AgnesImageProcessor
from gptqmodel.models.ports.agnes.modeling_agnes import (
    AgnesDynamicCache,
    AgnesForCausalLM,
    AgnesForConditionalGeneration,
    _delta_rule_chunked,
    _delta_rule_stepwise,
)
from gptqmodel.models.ports.agnes.processing_agnes import AgnesProcessor
from gptqmodel.models.ports.agnes.video_processing_agnes import AgnesVideoProcessor


def tiny_config(parallel=12, layer_types=None, bias=False):
    return AgnesConfig(
        text_config=AgnesTextConfig(
            vocab_size=64, hidden_size=16, num_hidden_layers=2, intermediate_size=32,
            parallel_ffn_intermediate_size=parallel, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, attention_bias=bias, linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3,
            layer_types=layer_types or ["agnes_delta_attention", "agnes_global_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
            pad_token_id=0, eos_token_id=None,
        ),
        vision_config=AgnesVisionConfig(
            depth=1, hidden_size=16, intermediate_size=24, num_heads=2, out_hidden_size=16,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=2, num_position_embeddings=16,
        ),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )


def test_definition_owns_model_selection():
    assert isinstance(AgnesQModel.loader, LocalModelLoader)
    assert not AgnesQModel.require_trust_remote_code


@pytest.mark.parametrize("seq", [1, 7, 67])
@torch.no_grad()
def test_delta_prefill_and_decode_match_independent_float64_recurrence(seq):
    torch.manual_seed(29)
    q, k = [torch.randn(2, seq, 2, 4) for _ in range(2)]
    v = torch.randn(2, seq, 2, 3)
    g = -torch.rand(2, seq, 2)
    beta = torch.rand(2, seq, 2)
    initial = torch.randn(2, 2, 4, 3)
    # A separate float64 token recurrence: decay, erase recalled value, write
    # gated residual, then read using the scaled query. Crosses the 64-token
    # chunk boundary, with nonzero initial state and unequal key/value widths.
    state = initial.double().clone()
    outputs = []
    for index in range(seq):
        key = k[:, index].double()
        query = q[:, index].double()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 2
        state = state * g[:, index].double().exp()[..., None, None]
        recalled = torch.einsum("bhk,bhkv->bhv", key, state)
        residual = beta[:, index].double()[..., None] * (v[:, index].double() - recalled)
        state = state + torch.einsum("bhk,bhv->bhkv", key, residual)
        outputs.append(torch.einsum("bhk,bhkv->bhv", query, state))
    expected = torch.stack(outputs, dim=1).float()
    for kernel in (_delta_rule_chunked, _delta_rule_stepwise):
        actual, actual_state = kernel(q, k, v, g, beta, initial_state=initial,
                                      output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_state, state.float(), rtol=2e-5, atol=2e-6)


def tiny_processor():
    vocab = {f"token_{i}": i for i in range(60)}
    special = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocab.update({token: i for i, token in enumerate(special, 60)})
    backend = Tokenizer(WordLevel(vocab, unk_token="token_0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="token_0",
                                       pad_token="token_0", additional_special_tokens=special)
    common = {"patch_size": 2, "temporal_patch_size": 1, "merge_size": 2,
                  "size": {"shortest_edge": 16, "longest_edge": 64}}
    return AgnesProcessor(AgnesImageProcessor(**common), tokenizer,
                          AgnesVideoProcessor(**common, min_frames=1, max_frames=4, do_sample_frames=False))


@pytest.mark.parametrize("kind", ["image", "video"])
def test_pixel_patch_order_matches_independent_indexing_oracle(kind):
    processor = tiny_processor()
    image = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)
    frames = torch.stack([image + i * 100 for i in range(3)])
    options = {"do_resize": False, "do_rescale": False, "do_normalize": False,
               "temporal_patch_size": 2, "return_tensors": "pt", "input_data_format": "channels_first"}
    if kind == "image":
        result = processor.image_processor(image, **options)
        frames = image[None].repeat(2, 1, 1, 1)
        pixels, grid = result.pixel_values, result.image_grid_thw
    else:
        result = processor.video_processor([frames], do_sample_frames=False, **options)
        frames = torch.cat((frames, frames[-1:]))
        pixels, grid = result.pixel_values_videos, result.video_grid_thw
    expected = []
    # Enumerate merged spatial blocks, their four patches, then each channel's
    # two temporal planes. This oracle deliberately uses no reshape/permute.
    for t in range(0, len(frames), 2):
        for block_w in range(0, 8, 4):
            for patch_h in (0, 2):
                for patch_w in (block_w, block_w + 2):
                    expected.append(torch.tensor([
                        frames[time, channel, h, w] for channel in range(3)
                        for time in (t, t + 1) for h in (patch_h, patch_h + 1)
                        for w in (patch_w, patch_w + 1)
                    ]))
    torch.testing.assert_close(pixels, torch.stack(expected), rtol=0, atol=0)
    assert grid.tolist() == [[len(frames) // 2, 2, 4]]


def test_processor_counts_and_metadata_do_not_leak_between_requests():
    processor = tiny_processor()
    image = torch.zeros(3, 4, 8, dtype=torch.uint8)
    frames = torch.zeros(3, 3, 4, 8, dtype=torch.uint8)
    counts = processor._get_num_multimodal_tokens(image_sizes=[(4, 8)], video_sizes=[(3, 4, 8)])
    image_patches = len(processor.image_processor(image).pixel_values)
    video_patches = len(processor.video_processor([frames]).pixel_values_videos)
    assert counts.num_image_patches == [image_patches]
    assert counts.num_image_tokens == [image_patches // 4]
    assert counts.num_video_tokens == [video_patches // 4]
    indices = [0, 1, 2]
    assert processor._frame_timestamps(indices, 2.0, 2) == [0.25, 1.0]
    assert indices == [0, 1, 2]
    default_size = dict(AgnesImageProcessor.size)
    AgnesImageProcessor(min_pixels=16, max_pixels=64)
    assert AgnesImageProcessor.size == default_size


@pytest.mark.parametrize("kind", ["image", "video"])
@torch.no_grad()
def test_local_processor_real_pixels_and_save_reload(tmp_path, kind):
    processor = tiny_processor()
    processor.save_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    restored = AgnesProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert type(restored.image_processor) is AgnesImageProcessor
    assert type(restored.video_processor) is AgnesVideoProcessor
    assert restored.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    if kind == "image":
        inputs = {"text": ["token_1 <|vision_start|><|image_pad|><|vision_end|>"],
                      "images": [torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(3, 4, 4)]}
    else:
        inputs = {"text": ["token_1 <|video_pad|>"],
                      "videos": [torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 4, 4)],
                      "video_metadata": [{"fps": 2.0, "total_num_frames": 2, "frames_indices": [0, 1]}]}
    batch = restored(**inputs, return_tensors="pt")
    expected = processor(**inputs, return_tensors="pt")
    for key in expected:
        torch.testing.assert_close(batch[key], expected[key])
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    output = model(**batch, use_cache=False)
    assert output.logits.shape[:2] == batch.input_ids.shape
    assert torch.isfinite(output.logits).all()
    # Tiny random weights may emit input-only modality placeholders. Restrict
    # this text-generation comparison to valid continuation tokens.
    generation = {"max_new_tokens": 2, "do_sample": False, "num_beams": 2,
                  "suppress_tokens": [60, 61, 62, 63]}
    generated = model.generate(**batch, **generation)
    uncached = model.generate(**batch, **generation, use_cache=False)
    assert torch.equal(generated, uncached)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
def test_processor_selection_ignores_malicious_json_and_preserves_overrides(tmp_path, monkeypatch, trust, bypass):
    processor = tiny_processor()
    processor.save_pretrained(tmp_path)
    tiny_config().save_pretrained(tmp_path)
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        path = tmp_path / name
        data = json.loads(path.read_text())
        data["auto_map"] = dict.fromkeys(("AutoConfig", "AutoProcessor", "AutoImageProcessor", "AutoVideoProcessor", "AutoTokenizer"), "poison.Evil")
        data["processor_class"] = "EvilProcessor"
        data["audio_tokenizer"] = {"audio_tokenizer_class": "EvilAudio", "audio_tokenizer_name_or_path": "poison"}
        path.write_text(json.dumps(data))
    (tmp_path / "poison.py").write_text("raise AssertionError('processor checkpoint Python executed')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    restored = AgnesProcessor.from_pretrained(tmp_path, trust_remote_code=trust)
    assert type(restored) is AgnesProcessor
    explicit = AgnesProcessor.from_pretrained(
        tmp_path, config=tiny_config(), tokenizer=processor.tokenizer,
        image_processor=processor.image_processor, video_processor=processor.video_processor,
        chat_template="explicit template", trust_remote_code=trust,
    )
    assert explicit.tokenizer is processor.tokenizer
    assert explicit.image_processor is processor.image_processor
    assert explicit.video_processor is processor.video_processor
    assert explicit.chat_template == "explicit template"


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.no_grad()
def test_public_load_owns_model_and_processor_with_malicious_metadata(tmp_path, monkeypatch, trust, bypass):
    from transformers import AutoProcessor

    original = AgnesForConditionalGeneration(tiny_config()).eval()
    original.save_pretrained(tmp_path)
    tiny_processor().save_pretrained(tmp_path)
    for name in ("config.json", "processor_config.json"):
        path = tmp_path / name
        data = json.loads(path.read_text())
        data["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModelForImageTextToText", "AutoProcessor", "AutoImageProcessor", "AutoVideoProcessor"), "poison.Evil")
        data["processor_class"] = "EvilProcessor"
        for component in ("image_processor", "video_processor"):
            if component in data:
                data[component]["auto_map"] = data["auto_map"]
        path.write_text(json.dumps(data))
    (tmp_path / "poison.py").write_text("raise AssertionError('public checkpoint Python executed')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")

    def reject_auto_processor(*args, **kwargs):
        raise AssertionError("local processor must not re-enter AutoProcessor")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", reject_auto_processor)
    loaded = GPTQModel.load(str(tmp_path), device="cpu", trust_remote_code=trust,
                           dtype=torch.float32, attn_implementation="eager")
    assert type(loaded.model) is AgnesForConditionalGeneration
    assert type(loaded.model.config) is AgnesConfig
    assert type(loaded.processor) is AgnesProcessor
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    batch = loaded.processor(
        text=["token_1 <|vision_start|><|image_pad|><|vision_end|>"],
        images=[torch.arange(48, dtype=torch.uint8).reshape(3, 4, 4)], return_tensors="pt",
    )
    torch.testing.assert_close(loaded.model(**batch, use_cache=False).logits,
                               original.to(dtype=loaded.model.dtype)(**batch, use_cache=False).logits)


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [False, True])
@torch.no_grad()
def test_malicious_checkpoint_never_selects_python(tmp_path, monkeypatch, trust, bypass):
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {
        "AutoConfig": "poison.EvilConfig",
        "AutoModelForImageTextToText": "poison.EvilModel",
    }
    path.write_text(json.dumps(config))
    # Authored sentinel code, deliberately inspected here; no external Python runs.
    (tmp_path / "poison.py").write_text("raise AssertionError('checkpoint Python executed')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1" if bypass else "0")
    loaded = AgnesQModel.loader.from_pretrained(tmp_path, trust_remote_code=trust)
    assert type(loaded) is AgnesForConditionalGeneration
    assert type(loaded.config) is AgnesConfig
    local_config = AgnesQModel.loader.load_config(tmp_path, trust_remote_code=trust)
    assert type(local_config) is AgnesConfig
    with torch.device("meta"):
        shell = AgnesQModel.loader.from_config(local_config, trust_remote_code=trust)
    assert type(shell) is AgnesForConditionalGeneration
    assert shell.model.visual.patch_embed.proj.weight.is_meta
    ids = torch.tensor([[1, 4, 7]])
    torch.testing.assert_close(loaded(ids, use_cache=False).logits, model(ids, use_cache=False).logits)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))


@pytest.mark.parametrize("parallel,bias", [(0, False), (12, True)])
@torch.no_grad()
def test_forward_cache_continuation_and_generation(parallel, bias):
    torch.manual_seed(17)
    model = AgnesForConditionalGeneration(tiny_config(parallel=parallel, bias=bias)).eval()
    ids = torch.tensor([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]])
    mask = ids.ne(0).long()
    full = model(ids, attention_mask=mask, use_cache=False).logits
    prefill = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    cache = prefill.past_key_values
    assert isinstance(cache, AgnesDynamicCache)
    assert cache.get_seq_length() == 3
    assert cache.has_previous_state(0)
    assert cache.layers[0].recurrent_states[0].abs().sum() > 0
    resumed = model(ids[:, 3:4], attention_mask=mask[:, :4], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(resumed.logits, full[:, 3:4], rtol=2e-5, atol=2e-6)
    resumed = model(ids[:, 4:], attention_mask=mask, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(resumed.logits, full[:, 4:], rtol=2e-5, atol=2e-6)
    assert cache.get_seq_length() == ids.shape[1]
    # Multi-token continuation takes the convolution/chunk route, not single-step decode.
    cache = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True).past_key_values
    chunk = model(ids[:, 3:], attention_mask=mask, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(chunk.logits, full[:, 3:], rtol=2e-5, atol=2e-6)
    generated = model.generate(ids[:1, :3], max_new_tokens=3, do_sample=False)
    uncached = model.generate(ids[:1, :3], max_new_tokens=3, do_sample=False, use_cache=False)
    assert generated.shape == (1, 6)
    assert torch.equal(generated, uncached)


@torch.no_grad()
def test_cache_batch_expansion_selection_and_crop_rejection():
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    ids = torch.tensor([[1, 3, 5], [2, 4, 6]])
    output = model(ids, use_cache=True)
    cache = output.past_key_values
    cache.rope_deltas = torch.zeros(2, 1, dtype=torch.long)
    cache.batch_repeat_interleave(2)
    assert cache.layers[0].conv_states[0].shape[0] == 4
    assert cache.rope_deltas.shape[0] == 4
    cache.batch_select_indices(torch.tensor([3, 0]))
    next_ids = torch.tensor([[7], [8]])
    actual = model(next_ids, past_key_values=cache, use_cache=True).logits
    expected = model(torch.cat((ids.flip(0), next_ids), dim=1), use_cache=False).logits[:, -1:]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    length = cache.get_seq_length()
    with pytest.raises(ValueError, match="cannot crop"):
        cache.crop(1)
    assert cache.get_seq_length() == length


@pytest.mark.parametrize("kind", ["image", "video"])
@torch.no_grad()
def test_multimodal_features_positions_generation_and_retry(kind):
    torch.manual_seed(19)
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    token = model.config.image_token_id if kind == "image" else model.config.video_token_id
    ids = torch.tensor([[1, 62, token, 63, 4]])
    mm_types = torch.tensor([[0, 0, 1 if kind == "image" else 2, 0, 0]])
    pixels = torch.randn(4, 12)  # t=1,h=w=2, each patch has 3*1*2*2 values
    grid = torch.tensor([[1, 2, 2]])
    kwargs = {"pixel_values" if kind == "image" else "pixel_values_videos": pixels,
              "image_grid_thw" if kind == "image" else "video_grid_thw": grid,
              "mm_token_type_ids": mm_types}
    mask = torch.ones_like(ids)
    output = model(ids, attention_mask=mask, **kwargs, use_cache=False)
    assert output.logits.shape == (1, 5, 64)
    assert output.rope_deltas.shape == (1, 1)
    modified = dict(kwargs)
    modified["pixel_values" if kind == "image" else "pixel_values_videos"] = pixels + 1
    changed = model(ids, attention_mask=mask, **modified, use_cache=False)
    assert not torch.allclose(output.logits[:, 2:], changed.logits[:, 2:])
    cached = model(ids, attention_mask=mask, **kwargs, use_cache=True)
    next_ids = torch.tensor([[7]])
    continuation = model(next_ids, attention_mask=torch.ones(1, 6),
                         past_key_values=cached.past_key_values, use_cache=True)
    combined = torch.cat((ids, next_ids), dim=1)
    full = model(combined, attention_mask=torch.ones_like(combined),
                 **{**kwargs, "mm_token_type_ids": torch.cat((mm_types, torch.zeros(1, 1, dtype=torch.long)), dim=1)},
                 use_cache=False)
    torch.testing.assert_close(continuation.logits, full.logits[:, -1:], rtol=2e-5, atol=2e-6)
    generated = model.generate(ids, attention_mask=mask, **kwargs, max_new_tokens=2, do_sample=False)
    uncached = model.generate(ids, attention_mask=mask, **kwargs, max_new_tokens=2, do_sample=False, use_cache=False)
    assert torch.equal(generated, uncached)
    # Failure on malformed placeholders must not poison the next valid request.
    with pytest.raises((ValueError, RuntimeError), match="features.*tokens|tokens.*features"):
        model(torch.tensor([[1, 2, 3, 4, 5]]), **kwargs, use_cache=False)
    retried = model(ids, attention_mask=mask, **kwargs, use_cache=False)
    torch.testing.assert_close(retried.logits, output.logits)


@pytest.mark.parametrize("layer_types", [["agnes_delta_attention"] * 2, ["agnes_global_attention"] * 2])
@torch.no_grad()
def test_homogeneous_plans_and_single_sample_left_padding(layer_types):
    model = AgnesForConditionalGeneration(tiny_config(layer_types=layer_types)).eval()
    ids = torch.tensor([[0, 0, 1, 3, 5]])
    mask = ids.ne(0).long()
    positions = mask.cumsum(-1).sub(1).clamp_min(0)
    padded = model(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, 2:]
    baseline = model(ids[:, 2:], use_cache=False).logits
    torch.testing.assert_close(padded, baseline, rtol=2e-5, atol=2e-6)
    full = model(ids, attention_mask=mask, use_cache=False).logits
    cache = model(ids[:, :3], attention_mask=mask[:, :3], use_cache=True).past_key_values
    output = model(ids[:, 3:], attention_mask=mask, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(output.logits, full[:, 3:], rtol=2e-5, atol=2e-6)
    assert cache.get_seq_length() == 5


@torch.no_grad()
def test_partial_cache_failure_requires_reset_before_retry():
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    ids = torch.tensor([[1, 3, 5]])
    reference = model(ids, use_cache=False).logits
    cache = AgnesDynamicCache(model.config.text_config)

    def fail(module, args):
        raise RuntimeError("injected later-layer failure")

    hook = model.model.language_model.layers[1].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected later-layer"):
        model(ids, past_key_values=cache, use_cache=True)
    hook.remove()
    assert cache.has_previous_state(0)  # first layer mutated before failure
    with pytest.raises(ValueError, match="incomplete forward.*reset"):
        model(ids, past_key_values=cache, use_cache=True)
    cache.reset()
    assert not cache.has_previous_state(0)
    retried = model(ids, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(retried.logits, reference)
    with pytest.raises(ValueError, match="only.*dynamic"):
        model.generate(ids, max_new_tokens=1, cache_implementation="static")


@torch.no_grad()
def test_multimodal_rope_offsets_belong_to_each_request_cache():
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    # Four merged visual tokens advance the spatial axes by only two: delta=-2.
    image_ids = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 4]])
    mm = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]])
    pixels = torch.randn(16, 12)
    image_kwargs = {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 4, 4]]), "mm_token_type_ids": mm}
    cache = model(image_ids, **image_kwargs, use_cache=True).past_key_values
    assert cache.rope_deltas.item() == -2
    # Interleave an unrelated request on the same model with a different offset.
    other = model(torch.tensor([[1, 62, 60, 63]]), pixel_values=pixels[:4],
                  image_grid_thw=torch.tensor([[1, 2, 2]]),
                  mm_token_type_ids=torch.tensor([[0, 0, 1, 0]]), use_cache=True)
    assert other.past_key_values.rope_deltas.item() == 0
    output = model(torch.tensor([[7]]), past_key_values=cache, use_cache=True).logits
    full = model(torch.cat((image_ids, torch.tensor([[7]])), dim=1),
                 **{**image_kwargs, "mm_token_type_ids": torch.cat((mm, torch.zeros(1, 1, dtype=torch.long)), dim=1)},
                 use_cache=False).logits[:, -1:]
    torch.testing.assert_close(output, full, rtol=2e-5, atol=2e-6)
    assert cache.rope_deltas.item() == -2


@pytest.mark.parametrize("head", ["causal", "conditional"])
@pytest.mark.parametrize("layout", ["delta", "global", "mixed"])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 63, 64, 65])
@pytest.mark.parametrize("kernel", [3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_short_prefill_and_kernel_boundaries(head, layout, length, kernel, dtype):
    torch.manual_seed(43)
    layers = {"delta": ["agnes_delta_attention"] * 2,
              "global": ["agnes_global_attention"] * 2,
              "mixed": ["agnes_delta_attention", "agnes_global_attention"]}[layout]
    config = tiny_config(layer_types=layers)
    config.text_config.linear_conv_kernel_dim = kernel
    model = (AgnesForCausalLM(config.text_config) if head == "causal"
             else AgnesForConditionalGeneration(config)).to(dtype=dtype).eval()
    ids = (torch.arange(length) % 40 + 1)[None]
    mask = torch.ones_like(ids)
    cached = model(ids, attention_mask=mask, use_cache=True)
    torch.testing.assert_close(cached.logits, model(ids, attention_mask=mask, use_cache=False).logits)
    # Single-token update followed by a multi-token convolution continuation.
    cache = cached.past_key_values
    for chunk in (torch.tensor([[7]]), torch.tensor([[9, 11]])):
        ids = torch.cat((ids, chunk), dim=1)
        actual = model(chunk, attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True).logits
        expected = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -chunk.shape[1]:]
        tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 0.03, "atol": 0.002}
        torch.testing.assert_close(actual, expected, **tolerance)
    for kind, layer in zip(layers, cache.layers):
        if kind == "agnes_delta_attention":
            assert layer.conv_states[0].shape[-1] == config.text_config.linear_conv_kernel_dim
    prompt = ids[:, :length]
    generated = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=2,
                               do_sample=False, suppress_tokens=[60, 61, 62, 63])
    expected = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=2,
                              do_sample=False, suppress_tokens=[60, 61, 62, 63], use_cache=False)
    assert torch.equal(generated, expected)


@pytest.mark.parametrize("head", ["causal", "conditional"])
@pytest.mark.parametrize("stage", ["head", "loss"])
@pytest.mark.parametrize("continuation", [False, True])
@torch.no_grad()
def test_public_head_or_loss_failure_invalidates_cache(head, stage, continuation, monkeypatch):
    torch.manual_seed(43)
    config = tiny_config()
    model = (AgnesForCausalLM(config.text_config) if head == "causal"
             else AgnesForConditionalGeneration(config)).eval()
    prefix = torch.tensor([[1, 3, 5]])
    ids = torch.tensor([[7]]) if continuation else prefix
    cache = model(prefix, use_cache=True).past_key_values if continuation else AgnesDynamicCache(config.text_config)
    in_progress = []

    def fail(*args, **kwargs):
        in_progress.append(cache._in_forward)
        raise RuntimeError("injected public-head failure")

    with monkeypatch.context() as patch:
        hook = None
        if stage == "head":
            hook = model.lm_head.register_forward_pre_hook(fail)
        else:
            patch.setattr(model, "_loss_function", fail, raising=False)
        try:
            with pytest.raises(RuntimeError, match="injected public-head failure"):
                model(ids, past_key_values=cache, use_cache=True, labels=ids if stage == "loss" else None)
        finally:
            if hook is not None:
                hook.remove()
    assert in_progress == [True], "cache must remain incomplete until the public head/loss succeeds"
    assert cache._in_forward
    with pytest.raises(ValueError, match="incomplete forward.*reset"):
        model(ids, past_key_values=cache, use_cache=True)
    cache.reset()
    assert cache.get_seq_length() == 0 and cache.rope_deltas is None
    full_ids = torch.cat((prefix, ids), dim=1) if continuation else ids
    retried = model(full_ids, labels=full_ids, past_key_values=cache, use_cache=True)
    expected = model(full_ids, labels=full_ids, use_cache=False)
    torch.testing.assert_close(retried.logits, expected.logits)
    torch.testing.assert_close(retried.loss, expected.loss)
    assert not cache._in_forward and cache.get_seq_length() == full_ids.shape[1]


def media_chunk(spec):
    """Independent token/grid fixture: each video frame has its own timestamp run."""
    ids, types = [1], [0]
    data = {}
    for kind, grid in spec:
        token, mm = (60, 1) if kind == "image" else (61, 2)
        t, h, w = grid
        for _ in range(t):
            count = h * w // 4
            ids += [62] + [token] * count + [63, 5]
            types += [0] + [mm] * count + [0, 0]
        pixel_key, grid_key = (("pixel_values", "image_grid_thw") if kind == "image"
                               else ("pixel_values_videos", "video_grid_thw"))
        data.setdefault(pixel_key, []).append(torch.randn(t * h * w, 12))
        data.setdefault(grid_key, []).append(torch.tensor([grid]))
    return torch.tensor([ids]), {**{key: torch.cat(values) for key, values in data.items()},
                                "mm_token_type_ids": torch.tensor([types])}


@pytest.mark.parametrize("kind", ["image", "video", "mixed"])
@pytest.mark.parametrize("prefix_kind", ["text", "media"])
@pytest.mark.parametrize("mask_mode", ["none", "full", "chunk"])
@pytest.mark.parametrize("explicit_positions", [False, True])
@torch.no_grad()
def test_appended_media_updates_positions_and_continues_text(kind, prefix_kind, mask_mode, explicit_positions):
    torch.manual_seed(43)
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    first, old = media_chunk([] if prefix_kind == "text" else [("image", [1, 4, 4])])
    if prefix_kind == "text":
        first, old["mm_token_type_ids"] = torch.tensor([[1, 3, 5]]), torch.zeros(1, 3, dtype=torch.long)
    spec = {"image": [("image", [1, 4, 4]), ("image", [1, 4, 6])],
            "video": [("video", [2, 4, 4]), ("video", [1, 4, 6])],
            "mixed": [("video", [2, 4, 4]), ("image", [1, 4, 6])]}[kind]
    later, new = media_chunk(spec)
    if mask_mode != "none":
        first = torch.cat((torch.zeros(1, 2, dtype=torch.long), first), dim=1)
        old["mm_token_type_ids"] = torch.cat((torch.zeros(1, 2, dtype=torch.long), old["mm_token_type_ids"]), dim=1)
    all_ids = torch.cat((first, later), dim=1)
    all_media = {key: torch.cat([data[key] for data in (old, new) if key in data], dim=0)
                 for key in old.keys() | new.keys() if key != "mm_token_type_ids"}
    all_media["mm_token_type_ids"] = torch.cat((old["mm_token_type_ids"], new["mm_token_type_ids"]), dim=1)
    mask = all_ids.ne(0).long() if mask_mode != "none" else None
    prefix_mask = mask[:, :first.shape[1]] if mask is not None else None
    current_mask = mask if mask_mode == "full" else torch.ones_like(later) if mask is not None else None
    cache = model(first, attention_mask=prefix_mask, **old, use_cache=True).past_key_values
    positions, delta = model.model.compute_3d_position_ids(
        later, model.get_input_embeddings()(later), attention_mask=current_mask, past_key_values=cache,
        **{key: value for key, value in new.items() if not key.startswith("pixel_values")},
    )
    full_positions, full_delta = model.model.get_rope_index(all_ids, attention_mask=mask,
        **{key: value for key, value in all_media.items() if not key.startswith("pixel_values")})
    torch.testing.assert_close(positions, full_positions[:, :, -later.shape[1]:])
    torch.testing.assert_close(delta, full_delta)
    assert full_delta.item() == -(5 if kind == "image" else 7) - (2 if prefix_kind == "media" else 0)
    extra = {"position_ids": full_positions[:, :, -later.shape[1]:]} if explicit_positions else {}
    actual = model(later, attention_mask=current_mask, **new, **extra, past_key_values=cache, use_cache=True)
    expected = model(all_ids, attention_mask=mask, **all_media, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits[:, -later.shape[1]:], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cache.rope_deltas, full_delta)
    text = torch.tensor([[7, 9]])
    complete = torch.cat((all_ids, text), dim=1)
    final_mask = complete.ne(0).long() if mask_mode == "full" else torch.ones_like(text) if mask is not None else None
    actual = model(text, attention_mask=final_mask, past_key_values=cache, use_cache=True).logits
    all_media["mm_token_type_ids"] = torch.cat((all_media["mm_token_type_ids"], torch.zeros_like(text)), dim=1)
    expected = model(complete, attention_mask=complete.ne(0).long() if mask is not None else None,
                     **all_media, use_cache=False).logits[:, -2:]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cache.rope_deltas, full_delta)


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("options", [
    {"size": {"shortest_edge": 16, "longest_edge": 16}},
    {"min_pixels": 16, "max_pixels": 16},
    {"min_pixels": 64},
    {"max_pixels": 16},
    {"size": {"shortest_edge": 16, "longest_edge": 256}, "min_pixels": 128},
    {"size": {"shortest_edge": 16, "longest_edge": 128}, "min_pixels": 16, "max_pixels": 32},
    {"size": {"shortest_edge": 16, "longest_edge": 16}, "do_resize": False},
    {"size": {"shortest_edge": 64, "longest_edge": 64}, "patch_size": 4, "merge_size": 1},
])
def test_patch_count_overrides_match_real_media_processing(kind, options):
    processor = tiny_processor()
    original = json.dumps(options, sort_keys=True)
    image = torch.zeros(3, 8, 8, dtype=torch.uint8)
    merge = options.get("merge_size", 2)
    for _ in range(2):
        if kind == "image":
            predicted = processor._get_num_multimodal_tokens(image_sizes=[(8, 8)], **options).num_image_tokens
            output = processor.image_processor([image], return_tensors="pt", **options)
            actual = output.image_grid_thw.prod(-1) // merge**2
        else:
            predicted = processor._get_num_multimodal_tokens(video_sizes=[(3, 8, 8)], **options).num_video_tokens
            output = processor.video_processor([image[None].repeat(3, 1, 1, 1)], return_tensors="pt", **options)
            actual = output.video_grid_thw.prod(-1) // merge**2
        assert predicted == actual.tolist()
    assert json.dumps(options, sort_keys=True) == original
    assert processor.image_processor.size == {"shortest_edge": 16, "longest_edge": 64}
    assert processor.video_processor.size == {"shortest_edge": 16, "longest_edge": 64}


@torch.no_grad()
def test_appended_media_padding_history_follows_beam_selection_and_reset():
    torch.manual_seed(43)
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    first = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 5], [0, 0, 0, 1, 62, 60, 63, 5]])
    types = first.eq(60).long()
    old_pixels = [torch.randn(16, 12), torch.randn(4, 12)]
    old_grids = torch.tensor([[1, 4, 4], [1, 2, 2]])
    mask = first.ne(0).long()
    cache = model(first, pixel_values=torch.cat(old_pixels), image_grid_thw=old_grids,
                  mm_token_type_ids=types, attention_mask=mask, use_cache=True).past_key_values
    mask.zero_()  # cache owns the mask snapshot, not a caller-mutable alias
    cache.batch_repeat_interleave(2)
    cache.batch_select_indices(torch.tensor([3, 0, 1]))
    selected = torch.tensor([1, 0, 0])
    first = first[selected]
    later = torch.tensor([[1, 62, 60, 60, 60, 60, 63, 5]]).repeat(3, 1)
    pixels = torch.randn(3, 16, 12)
    actual = model(later, pixel_values=pixels.flatten(0, 1), image_grid_thw=torch.tensor([[1, 4, 4]]).repeat(3, 1),
                   mm_token_type_ids=later.eq(60).long(), past_key_values=cache, use_cache=True).logits
    full = torch.cat((first, later), dim=1)
    full_pixels = torch.cat([part for i, index in enumerate(selected.tolist()) for part in (old_pixels[index], pixels[i])])
    full_grids = torch.cat([torch.stack((old_grids[index], torch.tensor([1, 4, 4]))) for index in selected.tolist()])
    kwargs = {"pixel_values": full_pixels, "image_grid_thw": full_grids,
              "mm_token_type_ids": full.eq(60).long(), "attention_mask": full.ne(0).long()}
    expected = model(full, **kwargs, use_cache=False).logits
    torch.testing.assert_close(actual, expected[:, -later.shape[1]:], rtol=2e-5, atol=2e-6)
    assert cache._agnes_valid_tokens.tolist() == [[13], [16], [16]]
    assert cache.rope_deltas.tolist() == [[-2], [-4], [-4]]
    cache.reset()
    assert cache._agnes_valid_tokens is None and cache._agnes_attention_mask is None
    replay = model(full, **kwargs, past_key_values=cache, use_cache=True)
    torch.testing.assert_close(replay.logits, expected)


@pytest.mark.parametrize("kind", ["image", "video", "mixed"])
@torch.no_grad()
def test_generate_appended_media_from_existing_cache(kind):
    torch.manual_seed(43)
    model = AgnesForConditionalGeneration(tiny_config()).eval()
    first, old = media_chunk([("image", [1, 4, 4])])
    spec = {"image": [("image", [1, 4, 6])], "video": [("video", [2, 4, 4])],
            "mixed": [("video", [2, 4, 4]), ("image", [1, 4, 6])]}[kind]
    later, new = media_chunk(spec)
    ids = torch.cat((first, later), dim=1)
    mm = torch.cat((old["mm_token_type_ids"], new["mm_token_type_ids"]), dim=1)
    cache = model(first, **old, use_cache=True).past_key_values
    full_kwargs = {key: torch.cat([data[key] for data in (old, new) if key in data])
                   for key in old.keys() | new.keys() if key != "mm_token_type_ids"}
    kwargs = {"attention_mask": torch.ones_like(ids), "mm_token_type_ids": mm,
              "max_new_tokens": 2, "do_sample": False, "suppress_tokens": [60, 61, 62, 63],
              "return_dict_in_generate": True, "output_logits": True}
    # Full input IDs plus a prefix cache, but only the new media's pixels/grids.
    actual = model.generate(ids, **{key: value for key, value in new.items() if key != "mm_token_type_ids"},
                            **kwargs, past_key_values=cache)
    expected = model.generate(ids, **full_kwargs, **kwargs, use_cache=False)
    assert torch.equal(actual.sequences, expected.sequences)
    for actual_logits, expected_logits in zip(actual.logits, expected.logits):
        torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-5, atol=2e-6)
