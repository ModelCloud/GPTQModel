# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer
from transformers import TokenizersBackend
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from gptqmodel.models.definitions.zdtaichu5 import ZDTaichu5QModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.zdtaichu5.configuration import ZDTaichu5_0_Config
from gptqmodel.models.ports.zdtaichu5.image_processing import ZDTaichu5_0_ImageProcessor
from gptqmodel.models.ports.zdtaichu5.modeling import ZDTaichu5_0_ForConditionalGeneration
from gptqmodel.models.ports.zdtaichu5.processing import ZDTaichu5_0_Processor
from tests.port_assets import read_port_asset, read_port_json


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/zdtaichu5/fixtures"


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(kv_heads=2, bias=False, layout="hybrid", attn="eager", ps_version="v2"):
    return ZDTaichu5_0_Config(
        vision_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
                           "image_size": 16, "patch_size": 4, "max_resolution": 32, "preferred_resolution": [16, 16],
                           "args": {"cpe_max_size": 32}, "mlp_ratio": 2},
        llm_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                        "num_attention_heads": 4, "num_key_value_heads": kv_heads, "head_dim": 8,
                        "vocab_size": 64, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                        "linear_key_head_dim": 8, "linear_value_head_dim": 8, "attention_bias": bias,
                        "layer_types": {"hybrid": ["linear_attention", "full_attention"],
                                     "full": ["full_attention"] * 2,
                                     "linear": ["linear_attention"] * 2}[layout],
                        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
                        "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                             "partial_rotary_factor": 1, "mrope_interleaved": True,
                                             "mrope_section": [1, 1, 2]}},
        force_image_size=16, vit_hidden_size=16, projector_hidden_size=32,
        img_context_token_id=60, video_context_token_id=61, ps_version=ps_version,
        attn_implementation=attn,
    )


def visual_inputs(batch=1):
    return {"input_ids": torch.tensor([[4, 60, 60, 60, 60, 5]]).expand(batch, -1).clone(),
                "pixel_values": torch.linspace(-1, 1, batch * 3 * 16 * 16).reshape(batch, 3, 16, 16),
                "image_grid_thw": torch.tensor([[1, 1, 1]]).expand(batch, -1).clone(),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0]]).expand(batch, -1).clone()}


def release_assets(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "processor_config.json"):
        (path / name).write_bytes(read_port_asset(FIXTURES / f"{name}"))


def public_config(**kwargs):
    config = tiny_config(**kwargs)
    config.llm_config.vocab_size = 248320
    config.llm_config.eos_token_id = config.eos_token_id = 248044
    config.llm_config.pad_token_id = config.pad_token_id = 248040
    config.img_context_token_id = 248056
    config.video_context_token_id = 248057
    return config


def test_definition_owns_model_and_media_dispatch():
    assert isinstance(ZDTaichu5QModel.loader, LocalModelLoader)
    assert not ZDTaichu5QModel.require_trust_remote_code
    assert ZDTaichu5QModel.fusable is None


def test_definition_declines_unqualified_awq_folding():
    assert "awq_get_modules_for_scaling" in ZDTaichu5QModel.__dict__
    assert ZDTaichu5QModel.awq_get_modules_for_scaling(None, None, None, None) == []


@pytest.mark.parametrize("layout", ["full", "linear", "hybrid"])
@pytest.mark.parametrize("kv_heads,bias,dtype", [
    (4, False, torch.float32), (2, True, torch.float16), (1, False, torch.bfloat16),
])
@torch.inference_mode()
def test_real_forward_cache_and_native_identity(layout, kv_heads, bias, dtype):
    torch.manual_seed(71)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config(kv_heads, bias, layout)).to(dtype).eval()
    assert type(model.language_model) is Qwen3_5ForCausalLM
    data = visual_inputs()
    data["pixel_values"] = data["pixel_values"].to(dtype)
    if layout == "linear":
        assert model(**data, use_cache=False).logits.isfinite().all()
        with pytest.raises(ValueError, match="at least one full-attention"):
            model(**data, use_cache=True)
        return
    output = model(**data, use_cache=True)
    assert output.logits.shape == (1, 6, 64) and output.logits.isfinite().all()
    continuation = torch.tensor([[7]])
    cached = model(input_ids=continuation, past_key_values=output.past_key_values, use_cache=True).logits
    full = {**data, "input_ids": torch.cat([data["input_ids"], continuation], -1),
            "mm_token_type_ids": torch.cat([data["mm_token_type_ids"], torch.zeros_like(continuation)], -1)}
    expected = model(**full, use_cache=False).logits[:, -1:]
    tolerance = {torch.float32: 2e-5, torch.float16: 5e-4, torch.bfloat16: 4e-3}[dtype]
    torch.testing.assert_close(cached, expected, atol=tolerance, rtol=tolerance)


@torch.inference_mode()
def test_generation_and_two_safetensors_cycles(tmp_path):
    torch.manual_seed(13)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config()).eval()
    data = visual_inputs()
    expected = model(**data, use_cache=False).logits
    greedy = model.generate(**data, max_new_tokens=3, do_sample=False)
    beam = model.generate(**data, max_new_tokens=3, do_sample=False, num_beams=3)
    for cycle in range(2):
        target = tmp_path / str(cycle)
        model.save_pretrained(target)
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model = ZDTaichu5_0_ForConditionalGeneration.from_pretrained(target, attn_implementation="eager").eval()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        torch.testing.assert_close(model(**data, use_cache=False).logits, expected, atol=0, rtol=0)
        assert torch.equal(model.generate(**data, max_new_tokens=3, do_sample=False), greedy)
        assert torch.equal(model.generate(**data, max_new_tokens=3, do_sample=False, num_beams=3), beam)


# sha256 of the released TaichuAI/ZDTaichu5.0-9B safetensors headers: json of sorted [key, shape] pairs.
RELEASED_INVENTORY_SHA256 = "c04081cd16b725697bd6d122242e83dc12808aec4bd2e3017e38a981fed27a74"


def test_full_release_inventory_and_config_isolation():
    payload = read_port_json(FIXTURES / "config.json")
    original = json.dumps(payload, sort_keys=True)
    config = ZDTaichu5_0_Config(**payload)
    assert json.dumps(payload, sort_keys=True) == original
    with torch.device("meta"):
        model = ZDTaichu5_0_ForConditionalGeneration(config)
    actual = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert len(actual) == 820
    assert actual["language_model.model.embed_tokens.weight"] == [248320, 4096]
    assert actual["language_model.lm_head.weight"] == [248320, 4096]
    assert actual["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"] == [8192, 4096]
    assert actual["language_model.model.layers.3.self_attn.q_proj.weight"] == [8192, 4096]
    canonical = json.dumps(sorted(actual.items()), separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == RELEASED_INVENTORY_SHA256


def test_real_pinned_bpe_and_image_video_processor(tmp_path):
    release_assets(tmp_path)
    tokenizer = TokenizersBackend.from_pretrained(tmp_path)
    source = Tokenizer.from_str((tmp_path / "tokenizer.json").read_text())
    for text in ["Hello world!", "中文 Taichu café 123\n", "<|vision_start|><|image_pad|><|vision_end|>"]:
        assert tokenizer.encode(text, add_special_tokens=False) == source.encode(text).ids
    config = public_config()
    image = ZDTaichu5_0_ImageProcessor(image_size=16, patch_size=4, max_num_tiles=2)
    proc = ZDTaichu5_0_Processor(image, tokenizer)
    proc.config = config
    pixels = Image.new("RGB", (16, 16), color=(70, 110, 190))
    data = proc(text="describe <|image_pad|>", images=[pixels], return_tensors="pt")
    assert data["pixel_values"].shape == (1, 3, 16, 16)
    assert data["input_ids"].eq(248056).sum() == 4
    frames = proc(text="describe <|video_pad|>", videos=[pixels, pixels], return_tensors="pt")
    assert frames["pixel_values"].shape == (2, 3, 16, 16)
    assert frames["input_ids"].eq(248056).sum() == 8
    assert not frames["input_ids"].eq(248057).any()
    proc.save_pretrained(tmp_path / "saved")
    restored = ZDTaichu5_0_Processor.from_pretrained(tmp_path / "saved")
    again = restored(text="describe <|video_pad|>", videos=[pixels, pixels], return_tensors="pt")
    for key, value in frames.items():
        torch.testing.assert_close(again[key], value, atol=0, rtol=0)


def test_video_before_image_keeps_visual_payload_order(tmp_path):
    release_assets(tmp_path)
    proc = ZDTaichu5_0_Processor(
        ZDTaichu5_0_ImageProcessor(image_size=16, patch_size=4, max_num_tiles=1),
        TokenizersBackend.from_pretrained(tmp_path),
    )
    image = Image.new("RGB", (16, 16), "red")
    frames = [Image.new("RGB", (16, 16), "blue"), Image.new("RGB", (16, 16), "green")]
    actual = proc(text="<|video_pad|> then <|image_pad|>", images=[image], videos=frames, return_tensors="pt")
    expected = proc.image_processor(images=[*frames, image], return_tensors="pt")["pixel_values"]
    torch.testing.assert_close(actual["pixel_values"], expected, atol=0, rtol=0)


@pytest.mark.parametrize("trust,bypass", [(None, None), (False, "0"), (True, "1")])
@torch.inference_mode()
def test_public_load_never_executes_hostile_model_tokenizer_processor(tmp_path, monkeypatch, trust, bypass):
    from gptqmodel import BACKEND, GPTQModel

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = ZDTaichu5_0_ForConditionalGeneration(public_config()).eval()
    model.save_pretrained(tmp_path)
    release_assets(tmp_path)
    image = ZDTaichu5_0_ImageProcessor(image_size=16, patch_size=4, max_num_tiles=1)
    image.save_pretrained(tmp_path)
    sentinel = tmp_path / "EXECUTED"
    hostile = "from pathlib import Path\nPath(" + repr(str(sentinel)) + ").touch()\nraise RuntimeError('checkpoint Python executed')\n"
    (tmp_path / "hostile.py").write_text(hostile)
    (tmp_path / "custom_generate").mkdir()
    (tmp_path / "custom_generate/generate.py").write_text(hostile)
    for name in ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"):
        p = tmp_path / name
        value = json.loads(p.read_text())
        value["auto_map"] = dict.fromkeys(("AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor"), "hostile.Evil")
        if name == "tokenizer_config.json":
            value["auto_map"]["AutoTokenizer"] = ["hostile.Evil", "hostile.Evil"]
            value["tokenizer_class"] = "EvilTokenizer"
        p.write_text(json.dumps(value))
    loaded = GPTQModel.load(str(tmp_path), device="cpu", backend=BACKEND.TORCH,
                           dtype=torch.float32, trust_remote_code=trust)
    assert type(loaded.model) is ZDTaichu5_0_ForConditionalGeneration
    assert type(loaded.processor) is ZDTaichu5_0_Processor
    tok = getattr(loaded.tokenizer, "tokenizer", loaded.tokenizer)
    assert type(tok) is TokenizersBackend
    prompt = "中文 Taichu café 123\n"
    oracle = Tokenizer.from_str((tmp_path / "tokenizer.json").read_text())
    assert tok.encode(prompt, add_special_tokens=False) == oracle.encode(prompt).ids
    data = loaded.processor(text="describe <|image_pad|>", images=[Image.new("RGB", (16, 16), "red")], return_tensors="pt")
    data["pixel_values"] = data["pixel_values"].float()
    assert loaded.model(**data).logits.isfinite().all()
    generated = loaded.generate(**data, max_new_tokens=1, do_sample=False)
    assert generated.shape[-1] == data["input_ids"].shape[-1] + 1
    assert not sentinel.exists()


def vision_oracle(model, pixels):
    """Independent float64 patch/ViT/downsample/projector equations, no module calls."""
    root = model.vision_model.radio_model.model
    generator = root.patch_generator
    pixels = pixels.double()
    patch = generator.patch_size
    rows, cols = pixels.shape[-2] // patch, pixels.shape[-1] // patch
    patches = torch.stack([
        pixels[:, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch].flatten(1)
        for r in range(rows) for c in range(cols)
    ], 1)

    def linear(x, layer):
        return F.linear(x, layer.weight.double(), layer.bias.double() if layer.bias is not None else None)

    def norm(x, layer):
        if isinstance(layer, torch.nn.Identity):
            return x
        return F.layer_norm(x, layer.normalized_shape, layer.weight.double(), layer.bias.double(), layer.eps)

    pos = generator.pos_embed.double().reshape(1, generator.num_rows, generator.num_cols, -1).permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=(rows, cols), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
    hidden = linear(patches, generator.embedder) + pos
    if generator.cls_token.token is not None:
        hidden = torch.cat([generator.cls_token.token.double().unsqueeze(0).expand(hidden.shape[0], -1, -1), hidden], 1)
    hidden = norm(hidden, generator.patch_normalizer)
    for block in root.blocks:
        normalized = norm(hidden, block.norm1)
        q, k, v = linear(normalized, block.attn.qkv).reshape(
            *normalized.shape[:2], 3, block.attn.num_heads, -1,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape_as(normalized)
        hidden = hidden + linear(attended, block.attn.proj)
        hidden = hidden + linear(F.gelu(linear(norm(hidden, block.norm2), block.mlp.fc1)), block.mlp.fc2)
    hidden = norm(hidden, root.norm)[:, generator.num_skip:]
    side = rows
    factor = round(1 / model.downsample_ratio)
    hidden = hidden.reshape(pixels.shape[0], side, side, -1)
    merged = torch.stack([
        torch.cat([hidden[:, r + dr, c + dc] for dr in range(factor) for dc in range(factor)], -1)
        for r in range(0, side, factor) for c in range(0, side, factor)
    ], 1)
    if model.ps_version == "v1":
        merged = merged.reshape(pixels.shape[0], side // factor, side // factor, -1).transpose(1, 2).flatten(1, 2)
    merged = merged / torch.sqrt(merged.square().mean(-1, keepdim=True) + model.mlp1[0].eps)
    merged = merged * model.mlp1[0].weight.double()
    return linear(torch.relu(linear(merged, model.mlp1[1])).square(), model.mlp1[3])


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@torch.inference_mode()
def test_independent_visual_equations_injection_and_spatial_positions(version, attn):
    torch.manual_seed(143)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config(ps_version=version, attn=attn)).eval()
    data = visual_inputs()
    expected = vision_oracle(model, data["pixel_values"])
    actual = model.extract_feature(data["pixel_values"])
    torch.testing.assert_close(actual.double(), expected, atol=2e-7, rtol=2e-5)
    embeddings = model.get_input_embeddings()(data["input_ids"]).clone()
    embeddings[data["input_ids"] == 60] = expected.float().reshape(-1, 32)
    captured = []
    handle = model.language_model.model.layers[0].register_forward_pre_hook(
        lambda _, args, kwargs: captured.append((args[0] if args else kwargs["hidden_states"]).clone()),
        with_kwargs=True,
    )
    try:
        model(**data, use_cache=False)
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0], embeddings, atol=2e-7, rtol=2e-5)
    positions, delta = model.get_rope_index(data["input_ids"], data["mm_token_type_ids"], data["image_grid_thw"])
    assert positions.tolist() == [[[0, 1, 1, 1, 1, 3]], [[0, 1, 1, 2, 2, 3]], [[0, 1, 2, 1, 2, 3]]]
    assert delta.tolist() == [[-2]]


@torch.inference_mode()
def test_ragged_beams_match_per_row_stepwise_logits_and_cache_isolation():
    torch.manual_seed(81)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config()).eval()
    model.mlp1[3].weight.mul_(100)
    ids = torch.tensor([[0, 4] + [60] * 12 + [5, 6], [0] * 8 + [6] + [60] * 4 + [7, 8, 9]])
    mask = ids != 0
    types = ids.eq(60).long()
    pixels = torch.randn(4, 3, 16, 16)
    batched = {"input_ids": ids, "attention_mask": mask.long(), "pixel_values": pixels,
                   "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 1]]), "mm_token_type_ids": types}
    _, expanded = model._expand_inputs_for_generation(expand_size=3, **batched)
    torch.testing.assert_close(expanded["pixel_values"], torch.cat([pixels[:3]] * 3 + [pixels[3:]] * 3), atol=0, rtol=0)
    actual = model.generate(**batched, max_new_tokens=3, num_beams=3,
                            do_sample=False, return_dict_in_generate=True, output_logits=True)
    for row in range(2):
        valid = mask[row]
        one = {"input_ids": ids[row:row + 1, valid], "attention_mask": torch.ones(1, valid.sum()),
                   "pixel_values": pixels[:3] if row == 0 else pixels[3:],
                   "image_grid_thw": torch.tensor([[1, 1, 2] if row == 0 else [1, 1, 1]]),
                   "mm_token_type_ids": types[row:row + 1, valid]}
        expected = model.generate(**one, max_new_tokens=3, num_beams=3,
                                  do_sample=False, return_dict_in_generate=True, output_logits=True)
        assert torch.equal(actual.sequences[row, -3:], expected.sequences[0, -3:])
        for observed, wanted in zip(actual.logits, expected.logits):
            torch.testing.assert_close(observed[row * 3:(row + 1) * 3], wanted, atol=2e-6, rtol=2e-5)
    first = model(**visual_inputs(), use_cache=True)
    other = visual_inputs()
    other["input_ids"] = torch.tensor([[3, 4, 5, 6, 7, 8]])
    model(input_ids=other["input_ids"], use_cache=True)
    cached = model(input_ids=torch.tensor([[7]]), past_key_values=first.past_key_values, use_cache=True)
    assert cached.logits.isfinite().all()


@pytest.mark.parametrize("layout,kv_heads,bias", [("hybrid", 2, True), ("full", 4, False), ("hybrid", 1, False)])
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_plain_awq_actual_pack_infer_and_two_exact_cycles(tmp_path, layout, kv_heads, bias, offload):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    torch.manual_seed(93)
    source = ZDTaichu5_0_ForConditionalGeneration(public_config(layout=layout, kv_heads=kv_heads, bias=bias)).eval()
    source.save_pretrained(tmp_path)
    release_assets(tmp_path)
    model = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(
        device="cpu", bits=4, group_size=16, fallback=None, offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offloaded"),
    ), backend=BACKEND.TORCH, dtype=torch.float16)
    image = Image.frombytes("RGB", (16, 16), bytes(range(256)) * 3)
    conversations = [[{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": "Describe these colors and textures in detail."},
    ]}]] * 2
    prepared = model.prepare_dataset(conversations, batch_size=1, calibration_data_min_length=1)[0]
    source = source.half()
    with torch.inference_mode():
        expected = source.get_input_embeddings()(prepared["input_ids"])
        vision = vision_oracle(source, prepared["pixel_values"].half())
        expected[prepared["input_ids"] == 248056] = vision.half().reshape(-1, 32)
    captured = []
    hook = model.model.language_model.model.layers[0].register_forward_pre_hook(
        lambda _, args, kwargs: captured.append((args[0] if args else kwargs["hidden_states"]).detach().clone()),
        with_kwargs=True,
    )
    dense = {k: v.half().clone() for k, v in source.state_dict().items()}
    try:
        model.quantize(CalibrationConfig(dataset=conversations, batch_size=1, min_length=1, sort="desc"), backend=BACKEND.TORCH)
    finally:
        hook.remove()
    assert captured
    torch.testing.assert_close(captured[0], expected, atol=0.003, rtol=0.003)
    packed_modules = {name for name, layer in model.model.named_modules() if isinstance(layer, AwqTorchLinear)}
    expected_modules = {
        f"language_model.model.layers.{i}.{name}" for i, layer in enumerate(source.language_model.model.layers)
        for name, item in layer.named_modules() if isinstance(item, torch.nn.Linear)
        and name not in {"linear_attn.in_proj_a", "linear_attn.in_proj_b"}
    }
    assert packed_modules == expected_modules
    for key, value in model.model.state_dict().items():
        if key in dense:
            torch.testing.assert_close(value, dense[key], atol=0, rtol=0)
    data = {**prepared, "pixel_values": prepared["pixel_values"].half()}
    with torch.inference_mode():
        expected_logits = model.model(**data, use_cache=False).logits
    assert expected_logits.isfinite().all()
    packed = {k: v.clone() for k, v in model.model.state_dict().items()}
    for cycle in range(2):
        target = tmp_path / f"quantized-{cycle}"
        model.save(str(target))
        model = GPTQModel.load(str(target), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
        assert {name for name, layer in model.model.named_modules() if isinstance(layer, AwqTorchLinear)} == expected_modules
        assert packed.keys() == model.model.state_dict().keys()
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value, packed[key], atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(model.model(**data, use_cache=False).logits, expected_logits, atol=0, rtol=0)
        assert list(target.glob("*.safetensors")) and not list(target.glob("*.bin"))


@pytest.mark.parametrize("assign", [False, True])
@torch.inference_mode()
def test_assignment_failure_repair_and_cache_preservation(tmp_path, assign):
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config()).eval()
    target = ZDTaichu5_0_ForConditionalGeneration(tiny_config()).eval()
    target.load_state_dict(model.state_dict(), strict=True, assign=assign)
    data = visual_inputs()
    expected = model(**data, use_cache=False).logits
    torch.testing.assert_close(target(**data, use_cache=False).logits, expected, atol=0, rtol=0)
    cache = target(input_ids=torch.tensor([[4, 5]]), use_cache=True).past_key_values
    length = cache.get_seq_length()
    with pytest.raises(ValueError, match="Visual feature count"):
        target(**{**data, "pixel_values": data["pixel_values"].repeat(2, 1, 1, 1)}, past_key_values=cache)
    assert cache.get_seq_length() == length
    assert target(input_ids=torch.tensor([[6]]), past_key_values=cache).logits.isfinite().all()
    model.save_pretrained(tmp_path)
    payload = (tmp_path / "model.safetensors").read_bytes()
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(Exception, match="[Hh]eader|[Ss]afetensor|[Dd]eserializ"):
        ZDTaichu5QModel.loader.from_pretrained(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(payload)
    repaired = ZDTaichu5QModel.loader.from_pretrained(tmp_path, attn_implementation="eager")
    torch.testing.assert_close(repaired(**data, use_cache=False).logits, expected, atol=0, rtol=0)


def test_processor_asset_failure_repair_and_pixel_oracle(tmp_path):
    release_assets(tmp_path)
    public_config().save_pretrained(tmp_path)
    config_path = tmp_path / "processor_config.json"
    payload = json.loads(config_path.read_text())
    payload["image_processor"]["norm_std"] = [0, 1, 1]
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="normalization"):
        ZDTaichu5_0_Processor.from_pretrained(tmp_path)
    payload["image_processor"]["norm_std"] = [0.229, 0.224, 0.225]
    config_path.write_text(json.dumps(payload))
    proc = ZDTaichu5_0_Processor.from_pretrained(tmp_path)
    image = Image.new("RGB", (16, 16), (70, 110, 190))
    actual = proc(text="<|image_pad|>", images=[image], return_tensors="pt")
    channels = (torch.tensor([70, 110, 190]) / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    expected = channels.bfloat16().reshape(1, 3, 1, 1).expand(1, 3, 16, 16)
    torch.testing.assert_close(actual["pixel_values"], expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="placeholder"):
        proc(text="no visual token", images=[image], return_tensors="pt")
    assert proc(text="<|image_pad|>", images=[image], return_tensors="pt")["input_ids"].eq(248056).sum() == 4


@torch.inference_mode()
def test_released_generation_policy_eos_sampling_and_two_cycles(tmp_path):
    from transformers import GenerationConfig

    torch.manual_seed(57)
    model = ZDTaichu5_0_ForConditionalGeneration(tiny_config()).eval()
    data = visual_inputs()
    release_policy = read_port_json(FIXTURES / "generation_config.json")
    policy = {**release_policy, "eos_token_id": [2, 3], "bos_token_id": 1, "pad_token_id": 0}
    model.generation_config = GenerationConfig(**policy)
    policy_before = model.generation_config.to_dict()
    torch.manual_seed(24)
    sample = model.generate(**data, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
    for cycle in range(2):
        path = tmp_path / str(cycle)
        model.save_pretrained(path)
        model = ZDTaichu5QModel.loader.from_pretrained(path, attn_implementation="eager")
        assert model.generation_config.to_dict() == policy_before
        torch.manual_seed(24)
        actual = model.generate(**data, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
        assert torch.equal(actual.sequences, sample.sequences)
        # Reloaded safetensors weights are mmapped at unaligned offsets, so the
        # single-token decode GEMV may differ by one ulp from the in-memory model.
        for observed, expected in zip(actual.scores, sample.scores):
            torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
    first = model.generate(**data, do_sample=False, max_new_tokens=1)[0, -1].item()
    caller = deepcopy(model.generation_config)
    caller.eos_token_id = [2, first]
    before = caller.to_dict()
    stopped = model.generate(**data, generation_config=caller, max_new_tokens=5, do_sample=False)
    assert stopped.shape[-1] == data["input_ids"].shape[-1] + 1
    assert caller.to_dict() == before and model.generation_config.to_dict() == policy_before
