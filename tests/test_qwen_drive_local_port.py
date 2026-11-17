# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Drive owns the released root VLM, never optional package registration."""

import builtins
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration

from gptqmodel import GPTQModel
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.qwen_drive import QwenDriveConfig, QwenDriveForConditionalGeneration
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.hf import load_model_config, resolve_trust_remote_code


FIXTURES = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/qwen_drive/fixtures"


def tiny_config(layout="hybrid", kv_heads=2, bias=False):
    layers = {
        "hybrid": ["linear_attention", "full_attention"],
        "full": ["full_attention", "full_attention"],
        "linear": ["linear_attention", "linear_attention"],
    }[layout]
    return QwenDriveConfig(
        vlm_config={
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": kv_heads,
                "head_dim": 8, "attention_bias": bias, "layer_types": layers,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
                "pad_token_id": 0, "eos_token_id": 3,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                    "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
            },
            "vision_config": {
                "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "num_position_embeddings": 16,
            },
            "image_token_id": 60, "video_token_id": 61, "vision_start_token_id": 62,
            "vision_end_token_id": 63,
        },
        expert_config={"model_type": "qwen_drive_planning_expert", "hidden_size": 1024},
    )


def native_reference(model):
    reference = Qwen3_5ForConditionalGeneration(copy.deepcopy(model.config.vlm_config)).to(model.dtype).eval()
    reference.load_state_dict({key.removeprefix("vlm."): value for key, value in model.state_dict().items()}, strict=True)
    return reference


def media_inputs(kind="image", dtype=torch.float32):
    video = kind == "video"
    token = 61 if video else 60
    return {
        "input_ids": torch.tensor([[1, 62, token, 63, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 2 if video else 1, 0, 0]]),
        "pixel_values_videos" if video else "pixel_values": torch.linspace(-1.7, 2.3, 96, dtype=dtype).reshape(4, 24),
        "video_grid_thw" if video else "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def test_definition_owns_model_without_official_registration():
    assert isinstance(QwenDriveQModel.loader, LocalModelLoader)
    assert QwenDriveQModel.loader.config_class == "gptqmodel.models.ports.qwen_drive.QwenDriveConfig"
    assert QwenDriveQModel.loader.model_class == "gptqmodel.models.ports.qwen_drive.QwenDriveForConditionalGeneration"
    assert QwenDriveQModel.require_trust_remote_code is False
    assert QwenDriveQModel.fusable is None


def test_entire_released_header_matches_meta_and_only_tied_head_is_omitted():
    data = (FIXTURES / "released_config.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == "91e765ca4a46e63b364b7e79f16bdb2b5d0e271684c566c03ae87d259c2d9ff8"
    header_data = gzip.decompress((FIXTURES / "released_header.json.gz").read_bytes())
    assert hashlib.sha256(header_data).hexdigest() == "03323c11647701e1a73546f00869740bd3e92261260f214714aa698cb656d54b"
    header = json.loads(header_data)
    header.pop("__metadata__")
    config = QwenDriveConfig(**json.loads(data))
    assert config.full_attention_layers == list(range(3, 32, 4))
    assert config.history_query_points == 15
    with torch.device("meta"):
        model = QwenDriveForConditionalGeneration(config)
    actual = {key: list(value.shape) for key, value in model.state_dict().items()}
    assert len(header) == 723
    assert actual.pop("vlm.lm_head.weight") == header["vlm.model.language_model.embed_tokens.weight"]["shape"]
    assert actual == {key: value["shape"] for key, value in header.items()}
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert not hasattr(model, "planning_expert")


@pytest.mark.parametrize("layout", ["hybrid", "full", "linear"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("bias", [False, True])
def test_native_exact_owner_and_forward_cache(layout, kv_heads, bias):
    model = QwenDriveForConditionalGeneration(tiny_config(layout, kv_heads, bias)).eval()
    reference = native_reference(model)
    ids = torch.tensor([[1, 7, 11, 13, 8, 9]])
    with torch.inference_mode():
        full = reference(ids, use_cache=False).logits
        torch.testing.assert_close(model(ids, use_cache=False).logits, full, rtol=0, atol=0)
        if layout == "linear":
            for candidate in (model, reference):
                with pytest.raises(ValueError, match="[Aa]ttention"):
                    candidate(ids[:, :1], use_cache=True)
            torch.testing.assert_close(model(ids, use_cache=False).logits, full, rtol=0, atol=0)
            return
        cache = None
        for start, end in [(0, 1), (1, 5), (5, 6)]:
            output = model(ids[:, start:end], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            torch.testing.assert_close(output.logits, full[:, start:end], rtol=1e-4, atol=2e-6)
    assert set(model.state_dict()) == {f"vlm.{key}" for key in reference.state_dict()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_media_native_parity_cache_and_greedy_beam(dtype, attention, kind):
    config = tiny_config()
    config._attn_implementation = attention
    model = QwenDriveForConditionalGeneration(config).to(dtype).eval()
    reference = native_reference(model)
    inputs = {"input_ids": torch.tensor([[1, 7, 11, 13, 8]])} if kind == "text" else media_inputs(kind, dtype)
    with torch.inference_mode():
        expected = reference(**inputs, use_cache=True)
        actual = model(**inputs, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for beams in (1, 3):
            options = {"max_new_tokens": 3, "do_sample": False, "num_beams": beams, "eos_token_id": None, "pad_token_id": 0}
            torch.testing.assert_close(model.generate(**inputs, **options), reference.generate(**inputs, **options))
        continuation = torch.tensor([[12]])
        torch.testing.assert_close(
            model(continuation, past_key_values=actual.past_key_values, use_cache=True).logits,
            reference(continuation, past_key_values=expected.past_key_values, use_cache=True).logits,
            rtol=0, atol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("assign", [False, True])
def test_dense_two_cycles_assignment_failure_repair_retry(tmp_path, dtype, assign):
    original = QwenDriveForConditionalGeneration._from_config(tiny_config(), dtype=dtype).eval()
    copied = QwenDriveForConditionalGeneration._from_config(tiny_config(), dtype=dtype).eval()
    copied.load_state_dict(original.state_dict(), assign=assign, strict=True)
    inputs = media_inputs(dtype=dtype)
    with torch.inference_mode():
        expected = original(**inputs, use_cache=False).logits
    for index in range(2):
        directory = tmp_path / str(index)
        copied.save_pretrained(directory, safe_serialization=True)
        assert not list(directory.glob("*.bin"))
        payload = directory / "model.safetensors"
        if index == 0:
            contents = payload.read_bytes()
            payload.write_bytes(b"corrupt")
            with pytest.raises(SafetensorError, match="[Hh]eader|[Ss]afetensor|[Dd]eserializ"):
                QwenDriveQModel.loader.from_pretrained(directory, dtype=dtype)
            payload.write_bytes(contents)
        copied = QwenDriveQModel.loader.from_pretrained(directory, dtype=dtype).eval()
        assert not hasattr(copied, "planning_expert")
        for name, value in original.state_dict().items():
            torch.testing.assert_close(copied.state_dict()[name], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(copied(**inputs, use_cache=False).logits, expected, rtol=0, atol=0)


def test_root_generation_policy_load_save_and_consumer(tmp_path):
    model = QwenDriveForConditionalGeneration(tiny_config()).eval()
    model.save_pretrained(tmp_path)
    GenerationConfig(bos_token_id=2, eos_token_id=[3, 9], pad_token_id=0, max_new_tokens=2,
                     do_sample=False, return_dict_in_generate=True, output_scores=True).save_pretrained(tmp_path)
    for index in range(2):
        model = QwenDriveQModel.loader.from_pretrained(tmp_path).eval()
        assert model.generation_config.eos_token_id == [3, 9]
        assert model.generation_config.bos_token_id == 2
        with torch.inference_mode():
            generated = model.generate(torch.tensor([[1, 7]]))
        assert 1 <= len(generated.scores) <= 2
        tmp_path = tmp_path / str(index)
        model.save_pretrained(tmp_path)


@pytest.mark.parametrize("where,foreign", [("root", "llama"), ("text_config", "qwen2"), ("vision_config", "clip")])
def test_wrong_layout_rejected_without_mutating_caller_config(where, foreign):
    config = tiny_config().vlm_config.to_dict()
    if where == "root":
        config["model_type"] = foreign
    else:
        config[where]["model_type"] = foreign
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="Qwen-Drive"):
        QwenDriveConfig(vlm_config=config)
    assert config == before


def write_checkpoint(path, layout="hybrid", kv_heads=2):
    model = QwenDriveForConditionalGeneration(tiny_config(layout, kv_heads)).eval()
    model.save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                                    "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                           bos_token="[BOS]", eos_token="[EOS]").save_pretrained(path)
    marker = path / "EXECUTED"
    sentinel = f"from pathlib import Path\nPath({str(marker)!r}).write_text('UNSAFE')\nraise RuntimeError('checkpoint Python')\n"
    for filename in ("configuration_hostile.py", "modeling_hostile.py", "tokenization_hostile.py"):
        (path / filename).write_text(sentinel)
    (path / "custom_generate").mkdir()
    (path / "custom_generate/generate.py").write_text(sentinel)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {"AutoConfig": "configuration_hostile.HostileConfig", "AutoModel": "modeling_hostile.HostileModel"}
    config["vlm_config"]["auto_map"] = dict(config["auto_map"])
    config_path.write_text(json.dumps(config))
    tokenizer_path = path / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_path.read_text())
    tokenizer_config["auto_map"] = {"AutoTokenizer": ["tokenization_hostile.HostileTokenizer", None]}
    tokenizer_path.write_text(json.dumps(tokenizer_config))
    return model, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_public_hostile_maps_never_import_official_or_checkpoint_python(tmp_path, monkeypatch, trust, bypass):
    source = tmp_path / "source"
    expected, marker = write_checkpoint(source)
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    original_import = builtins.__import__

    def deny_official(name, *args, **kwargs):
        if name == "qwen_drive" or name.startswith("qwen_drive."):
            raise AssertionError("optional official package must never be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_official)
    assert type(load_model_config(source, trust_remote_code=trust)) is QwenDriveConfig
    assert resolve_trust_remote_code(source, trust_remote_code=trust) is False
    loaded = GPTQModel.load(str(source), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded.model) is QwenDriveForConditionalGeneration
    assert loaded.trust_remote_code is False
    inputs = media_inputs()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**inputs, use_cache=False).logits, expected(**inputs, use_cache=False).logits,
                                   rtol=0, atol=0)
        assert loaded.generate(**inputs, max_new_tokens=2, do_sample=False).shape == (1, 7)
    assert not marker.exists()


@pytest.mark.parametrize("layout", ["hybrid", "full"])
@pytest.mark.parametrize("offload", [False, True])
def test_public_image_plain_awq_all_linears_and_two_exact_reloads(tmp_path, monkeypatch, layout, offload):
    source = tmp_path / "source"
    original, marker = write_checkpoint(source, layout)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=offload,
                     offload_to_disk_path=str(tmp_path / "offload") if offload else None)
    loaded = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    assert loaded.awq_get_modules_for_scaling(None, None, None) == []
    inputs = media_inputs(dtype=torch.float16)
    reference = native_reference(original).half()
    first_inputs = []

    def observe_first(module, args, kwargs):
        first_inputs.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    handle = reference.model.language_model.layers[0].register_forward_pre_hook(observe_first, with_kwargs=True)
    try:
        with torch.inference_mode():
            reference(**inputs, use_cache=False)
    finally:
        handle.remove()
    assert len(first_inputs) == 1
    image_position = inputs["input_ids"] == 60
    assert not torch.equal(first_inputs[0][image_position], reference.get_input_embeddings()(inputs["input_ids"])[image_position])
    captured = []
    cache_inputs = StageInputsCapture.cache_inputs

    def check_capture(stage, *args, **kwargs):
        cache = cache_inputs(stage, *args, **kwargs)
        assert len(cache.layer_inputs) == 1
        actual = cache.layer_inputs[0][0]
        torch.testing.assert_close(actual.cpu(), first_inputs[0].cpu(), rtol=0, atol=0)
        captured.append(actual.detach().clone())
        return cache

    monkeypatch.setattr(StageInputsCapture, "cache_inputs", check_capture)
    loaded.quantize(CalibrationConfig(dataset=[inputs], batch_size=1, min_length=1), backend=BACKEND.TORCH)
    assert len(captured) == 1
    expected = {
        f"vlm.model.language_model.layers.{index}.{projection}"
        for index, kind in enumerate(original.config.vlm_config.text_config.layer_types)
        for projection in (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                           if kind == "full_attention" else
                           ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"])
        + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    }
    assert {name for name, module in loaded.model.named_modules() if isinstance(module, AwqTorchLinear)} == expected
    state = {key: value.clone() for key, value in loaded.model.state_dict().items()}
    for key, value in original.state_dict().items():
        if not any(key.startswith(prefix + ".") for prefix in expected):
            torch.testing.assert_close(state[key], value.half(), rtol=0, atol=0)
    with torch.inference_mode():
        logits = loaded(**inputs, use_cache=False).logits
    assert torch.isfinite(logits).all()
    for cycle in range(2):
        destination = tmp_path / f"packed-{cycle}"
        loaded.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        loaded = GPTQModel.load(str(destination), backend=BACKEND.TORCH, dtype=torch.float16, device="cpu")
        assert type(loaded.model) is QwenDriveForConditionalGeneration
        assert set(loaded.model.state_dict()) == set(state)
        for name, value in state.items():
            torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded(**inputs, use_cache=False).logits, logits, rtol=0, atol=0)
    assert not marker.exists()


def test_public_subfolder_failed_config_repair_retry(tmp_path):
    path = tmp_path / "nested"
    expected, marker = write_checkpoint(path)
    config_file = path / "config.json"
    contents = config_file.read_bytes()
    config_file.write_text("{bad-json")
    with pytest.raises(OSError, match="JSON"):
        GPTQModel.load(str(tmp_path), subfolder="nested", device="cpu", dtype=torch.float32)
    config_file.write_bytes(contents)
    loaded = GPTQModel.load(str(tmp_path), subfolder="nested", trust_remote_code=True, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        inputs = media_inputs()
        torch.testing.assert_close(loaded(**inputs, use_cache=False).logits, expected(**inputs, use_cache=False).logits,
                                   rtol=0, atol=0)
    assert not marker.exists()


@pytest.mark.parametrize("container", [list, tuple, iter, lambda rows: rows[0]])
def test_native_media_calibration_preserves_geometry_and_caller_rows(container):
    wrapper = QwenDriveQModel(QwenDriveForConditionalGeneration(tiny_config()), quantized=False,
                              quantize_config=AWQConfig(device="cpu", group_size=16))
    row = media_inputs()
    prepared = wrapper.prepare_dataset(container([row]), calibration_data_min_length=1)
    assert len(prepared) == 1 and prepared[0] is not row
    assert set(prepared[0]) == set(row)
    for key, value in row.items():
        torch.testing.assert_close(prepared[0][key], value, rtol=0, atol=0)
    assert wrapper.prepare_dataset([row], calibration_data_min_length=5) == []
    for options in ({"batch_size": 2}, {"calibration_dataset_concat_size": 10}):
        with pytest.raises(ValueError, match="batch_size|concatenat"):
            wrapper.prepare_dataset([row], **options)
    malformed = dict(row)
    malformed["input_ids"] = row["input_ids"][0]
    with pytest.raises(ValueError, match="batched"):
        wrapper.prepare_dataset([malformed])
    with pytest.raises(ValueError, match="dictionaries"):
        wrapper.prepare_dataset([row, "drop this image"])


REAL_TOKEN_CASES = [
    ("Hello world!", [9419, 1814, 0], "Hello world!"),
    ("Planning: left turn at 10 m/s.", [80116, 25, 2047, 2404, 506, 220, 16, 15, 295, 2600, 13], "Planning: left turn at 10 m/s."),
    ("转向灯亮起，行人正在过马路。", [144191, 138114, 3709, 102464, 99443, 140643, 1710], "转向灯亮起，行人正在过马路。"),
    (" e\u0301 café ١٢٣\n\t  ", [3825, 50203, 220, 149, 94, 149, 95, 149, 96, 198, 8424], " é café ١٢٣\n\t  "),
    ("<|vision_start|><|image_pad|><|vision_end|>", [248053, 248056, 248054], "<|vision_start|><|image_pad|><|vision_end|>"),
]


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_real_released_tokenizer_public_ids_chat_media_and_two_reloads(tmp_path, monkeypatch, trust, bypass):
    # IDs are pinned from the released Rust tokenizer, not from the loader under test.
    config = tiny_config()
    config.vlm_config.text_config.vocab_size = 248077
    config.vlm_config.text_config.eos_token_id = 248046
    config.vlm_config.text_config.pad_token_id = 248044
    config.vlm_config.image_token_id = 248056
    config.vlm_config.video_token_id = 248057
    config.vlm_config.vision_start_token_id = 248053
    config.vlm_config.vision_end_token_id = 248054
    source = tmp_path / "source"
    model = QwenDriveForConditionalGeneration._from_config(config, dtype=torch.float16).eval()
    model.save_pretrained(source)
    manifest = json.loads((FIXTURES / "provenance.json").read_text())
    for name, entry in manifest["assets"].items():
        data = gzip.decompress((FIXTURES / (name + ".gz")).read_bytes())
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        (source / name).write_bytes(data)
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    tokenizer_config["auto_map"] = {"AutoTokenizer": ["hostile.HostileTokenizer", None]}
    (source / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    model_config = json.loads((source / "config.json").read_text())
    model_config["auto_map"] = {"AutoConfig": "hostile.HostileConfig", "AutoModel": "hostile.HostileModel"}
    (source / "config.json").write_text(json.dumps(model_config))
    marker = source / "EXECUTED"
    (source / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint Python')\n")
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    inputs = media_inputs(dtype=torch.float16)
    inputs["input_ids"] = torch.tensor([[1, 248053, 248056, 248054, 5]])
    with torch.inference_mode():
        expected_logits = model(**inputs, use_cache=False).logits
    expected_chat = "<|im_start|>user\nWhat should the ego car do?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    expected_chat_ids = [248045, 846, 198, 3710, 1220, 279, 35075, 1740, 635, 30, 248046, 198, 248045, 74455, 198, 248068, 198]
    for cycle in range(3):
        loaded = GPTQModel.load(str(source), trust_remote_code=trust, device="cpu", dtype=torch.float16)
        assert loaded.trust_remote_code is False
        for text, expected_ids, decoded in REAL_TOKEN_CASES:
            assert loaded.tokenizer.encode(text) == expected_ids
            assert loaded.tokenizer.decode(expected_ids, skip_special_tokens=False) == decoded
        chat = loaded.tokenizer.apply_chat_template([{"role": "user", "content": "What should the ego car do?"}],
                                                    tokenize=False, add_generation_prompt=True)
        assert chat == expected_chat
        assert loaded.tokenizer.encode(chat) == expected_chat_ids
        assert loaded.tokenizer.eos_token_id == 248046 and loaded.tokenizer.pad_token_id == 248044
        with torch.inference_mode():
            torch.testing.assert_close(loaded(**inputs, use_cache=False).logits, expected_logits, rtol=0, atol=0)
            if cycle == 0:
                assert loaded.generate(**inputs, max_new_tokens=1, do_sample=False).shape == (1, 6)
        if cycle < 2:
            destination = tmp_path / str(cycle)
            loaded.model.save_pretrained(destination, safe_serialization=True)
            loaded.tokenizer.save_pretrained(destination)
            source = destination
    assert not marker.exists()
