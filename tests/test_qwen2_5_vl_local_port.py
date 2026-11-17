# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native Qwen2.5-VL decoder and explicit text-head qualification."""

import copy
import json

import pytest
import torch
from torch import nn
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextConfig,
)

from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel


def text_config(kv_heads=2, **kwargs):
    return Qwen2_5_VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    )


def composite_config():
    return Qwen2_5_VLConfig(
        text_config=text_config().to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )


def test_text_alias_has_explicit_head_and_decoder_tree():
    config = text_config()
    model = Qwen2_5_VLQModel.loader.from_config(config, attn_implementation="eager").eval()
    ids = torch.tensor([[1, 4, 9, 3]])
    with torch.no_grad():
        output = model(ids, use_cache=True)
    assert output.logits.shape == (1, 4, 64)
    assert output.past_key_values.get_seq_length() == 4
    tree = Qwen2_5_VLQModel._resolve_effective_module_tree(model, None)
    with Qwen2_5_VLQModel._module_tree_context(tree):
        assert Qwen2_5_VLQModel.extract_layers_node() == ["model.layers"]


def test_composite_remains_exact_native_class():
    model = Qwen2_5_VLQModel.loader.from_config(composite_config(), attn_implementation="eager")
    assert type(model) is Qwen2_5_VLForConditionalGeneration


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_text_native_decoder_head_equation(kv_heads):
    model = Qwen2_5_VLQModel.loader.from_config(text_config(kv_heads), attn_implementation="eager").eval()
    ids = torch.tensor([[1, 4, 9, 3], [1, 8, 5, 7]])
    with torch.no_grad():
        expected = torch.nn.functional.linear(model.model(ids).last_hidden_state, model.lm_head.weight)
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def make_text_model(kv_heads=2, dtype=torch.float32, implementation="eager", **kwargs):
    return Qwen2_5_VLQModel.loader.from_config(
        text_config(kv_heads, **kwargs),
        attn_implementation=implementation,
        dtype=dtype,
    ).eval()


def save_tokenizer(path):
    # Synthetic vocabulary for tiny model loading, not a released-tokenizer oracle.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("sliding", [False, True])
def test_text_matches_native_composite_decoder_cache_and_generation(dtype, implementation, sliding):
    config = composite_config()
    config.text_config.use_sliding_window = sliding
    config.text_config.sliding_window = 3 if sliding else None
    config.text_config.layer_types = ["full_attention", "sliding_attention"] if sliding else ["full_attention"] * 2
    text = Qwen2_5_VLQModel.loader.from_config(
        copy.deepcopy(config.text_config),
        dtype=dtype,
        attn_implementation=implementation,
    ).eval()
    native = Qwen2_5_VLForConditionalGeneration._from_config(
        config, dtype=dtype, attn_implementation=implementation
    ).eval()
    native.model.language_model.load_state_dict(text.model.state_dict())
    native.lm_head.load_state_dict(text.lm_head.state_dict())
    ids = torch.tensor([[0, 1, 5, 8, 7], [1, 6, 4, 9, 3]])
    mask = (ids != 0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    with torch.inference_mode():
        expected = native(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        actual = text(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
        suffix = torch.tensor([[4, 6], [8, 5]])
        continued_mask = torch.cat([mask, torch.ones(2, 2, dtype=torch.long)], dim=1)
        continued_positions = continued_mask.cumsum(-1)[:, -2:] - 1
        expected_next = native(
            suffix,
            attention_mask=continued_mask,
            position_ids=continued_positions,
            past_key_values=expected.past_key_values,
            use_cache=True,
        )
        actual_next = text(
            suffix,
            attention_mask=continued_mask,
            position_ids=continued_positions,
            past_key_values=actual.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(actual_next.logits, expected_next.logits, atol=0, rtol=0)
        assert actual_next.past_key_values.get_seq_length() == 7
        for beams in (1, 3):
            args = {
                "attention_mask": mask,
                "max_new_tokens": 3,
                "do_sample": False,
                "num_beams": beams,
                "eos_token_id": None,
            }
            torch.testing.assert_close(text.generate(ids, **args), native.generate(ids, **args), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied", [False, True])
def test_text_safetensors_two_cycles_assignment_and_failure_repair(tmp_path, dtype, tied):
    model = make_text_model(dtype=dtype, tie_word_embeddings=tied)
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        expected = model(ids).logits
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for assign in (False, True):
        target = make_text_model(dtype=dtype, tie_word_embeddings=tied)
        target.load_state_dict(snapshot, assign=assign)
        with torch.inference_mode():
            torch.testing.assert_close(target(ids).logits, expected, rtol=0, atol=0)
    for cycle in range(2):
        path = tmp_path / str(cycle)
        model.save_pretrained(path, safe_serialization=True)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        from gptqmodel.utils.hf import load_model_config

        config = load_model_config(path, trust_remote_code=True)
        model = Qwen2_5_VLQModel.loader.from_pretrained(
            path, config=config, dtype=dtype, attn_implementation="eager"
        ).eval()
        assert isinstance(model.lm_head, nn.Linear)
        assert (model.lm_head.weight is model.model.embed_tokens.weight) is tied
        assert model.state_dict().keys() == snapshot.keys()
        for key, value in model.state_dict().items():
            assert value.dtype == snapshot[key].dtype
            torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).logits, expected, atol=0, rtol=0)
    broken_path = tmp_path / "1" / "config.json"
    original = broken_path.read_bytes()
    broken_path.write_text('{"model_type":"foreign"}')
    with pytest.raises(ValueError, match="Unsupported Qwen2.5-VL"):
        Qwen2_5_VLQModel.loader.load_config(tmp_path / "1")
    broken_path.write_bytes(original)
    repaired = Qwen2_5_VLQModel.loader.from_pretrained(tmp_path / "1", dtype=dtype, attn_implementation="eager").eval()
    with torch.inference_mode():
        torch.testing.assert_close(repaired(ids).logits, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bypass", [None, "0", "1"])
@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("text_only", [False, True])
def test_public_hostile_auto_map_ignores_remote_code(tmp_path, monkeypatch, bypass, trust, text_only):
    from gptqmodel import GPTQModel
    from gptqmodel.utils import BACKEND

    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    model = (
        make_text_model()
        if text_only
        else Qwen2_5_VLForConditionalGeneration._from_config(
            composite_config(),
            attn_implementation="eager",
        ).eval()
    )
    model.save_pretrained(tmp_path)
    save_tokenizer(tmp_path)
    if not text_only:
        from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor

        Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
            tokenizer=PreTrainedTokenizerFast.from_pretrained(tmp_path),
            video_processor=Qwen2VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        ).save_pretrained(tmp_path)
    marker = tmp_path / "EXECUTED"
    (tmp_path / "evil.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('evil')\n"
    )
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    payload["auto_map"] = dict.fromkeys(
        ("AutoConfig", "AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText"), "evil.Evil"
    )
    config_path.write_text(json.dumps(payload))
    loaded = GPTQModel.load(
        str(tmp_path), backend=BACKEND.TORCH, dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    assert loaded.trust_remote_code is False
    assert type(loaded.model) is type(model)
    assert loaded.require_load_processor is (not text_only)
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=0, rtol=0)
    assert not marker.exists()


def test_text_head_outputs_loss_selection_and_media_rejection():
    model = make_text_model()
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        output = model(ids, labels=ids, output_hidden_states=True, output_attentions=True)
        expected_loss = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)
        )
        torch.testing.assert_close(output.loss, expected_loss)
        torch.testing.assert_close(model(ids, logits_to_keep=2).logits, output.logits[:, -2:])
        torch.testing.assert_close(model(ids, logits_to_keep=torch.tensor([0, 2])).logits, output.logits[:, [0, 2]])
        torch.testing.assert_close(model(ids, return_dict=False)[0], output.logits)
        assert len(output.hidden_states) == 3 and len(output.attentions) == 2
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
            "second_per_grid_ts",
            "mm_encoder_outputs",
        ):
            with pytest.raises(ValueError, match="text only"):
                model(ids, **{key: torch.ones(4, 24)})
            torch.testing.assert_close(model(ids).logits, output.logits)


@pytest.mark.parametrize("beams", [1, 3])
def test_native_composite_media_forward_and_generation_unchanged(beams):
    config = composite_config()
    model = Qwen2_5_VLQModel.loader.from_config(config, attn_implementation="eager").eval()
    native = Qwen2_5_VLForConditionalGeneration._from_config(copy.deepcopy(config), attn_implementation="eager").eval()
    native.load_state_dict(model.state_dict())
    # One image and one two-frame video, each producing one merged patch.
    ids = torch.tensor([[1, 62, 60, 63, 4, 62, 61, 63, 5]])
    arguments = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0, 0, 2, 0, 0]]),
        "pixel_values": torch.linspace(-1, 1, 96).reshape(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "pixel_values_videos": torch.linspace(1, -1, 96).reshape(4, 24),
        "video_grid_thw": torch.tensor([[1, 2, 2]]),
        "second_per_grid_ts": torch.tensor([1.0]),
    }
    with torch.inference_mode():
        torch.testing.assert_close(model(**arguments).logits, native(**arguments).logits, atol=0, rtol=0)
        expected = native.generate(**arguments, max_new_tokens=2, do_sample=False, num_beams=beams, eos_token_id=None)
        actual = model.generate(**arguments, max_new_tokens=2, do_sample=False, num_beams=beams, eos_token_id=None)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_text_generation_metadata_survives_two_native_save_cycles(tmp_path):
    model = make_text_model()
    model.generation_config.eos_token_id = [2, 7]
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.73
    model.generation_config.top_k = 9
    model.generation_config.top_p = 0.81
    for index in range(2):
        destination = tmp_path / str(index)
        model.save_pretrained(destination)
        model = Qwen2_5_VLQModel.loader.from_pretrained(destination, attn_implementation="eager")
        assert model.generation_config.eos_token_id == [2, 7]
        assert model.generation_config.do_sample is True
        assert model.generation_config.temperature == 0.73
        assert model.generation_config.top_k == 9
        assert model.generation_config.top_p == 0.81


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_public_plain_awq_packs_all_linears_and_reloads_twice(tmp_path, offload, kv_heads):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig

    source = tmp_path / "source"
    make_text_model(kv_heads).save_pretrained(source)
    save_tokenizer(source)
    qcfg = AWQConfig(
        device="cpu",
        group_size=16,
        fallback=None,
        offload_to_disk=offload,
        offload_to_disk_path=str(tmp_path / "offload"),
    )
    wrapped = GPTQModel.load(
        str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
    )
    calibration = [
        {"input_ids": torch.tensor([[1, 4, 5, 6, 7, 4, 6, 5]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)},
        {"input_ids": torch.tensor([[1, 7, 6, 4, 5, 7, 4, 6]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)},
    ]
    # Source tensors, independently loaded before quantization, are the dense
    # exclusion oracle even when the public model is initially a meta shell.
    dense = Qwen2_5_VLQModel.loader.from_pretrained(source, dtype=torch.float16, attn_implementation="eager")
    excluded = {
        key: value.detach().clone()
        for key, value in dense.state_dict().items()
        if any(token in key for token in ("embed_tokens", "norm", "lm_head"))
    }
    del dense
    assert wrapped.extract_layers_node() == ["model.layers"]
    wrapped.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == {
        f"model.layers.{index}.{projection}"
        for index in range(2)
        for projection in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    for key, value in excluded.items():
        torch.testing.assert_close(wrapped.model.state_dict()[key], value, atol=0, rtol=0)
    with torch.inference_mode():
        expected = wrapped.model(**calibration[0], use_cache=False).logits
        assert torch.isfinite(expected).all()
    state = {key: value.detach().clone() for key, value in wrapped.model.state_dict().items()}
    for cycle in range(2):
        destination = tmp_path / f"packed-{cycle}"
        wrapped.save(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        wrapped = GPTQModel.load(
            str(destination), backend=BACKEND.TORCH, dtype=torch.float16, attn_implementation="eager"
        )
        assert wrapped.model.state_dict().keys() == state.keys()
        for key, value in state.items():
            actual = wrapped.model.state_dict()[key]
            assert actual.dtype == value.dtype
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(
                wrapped.model(**calibration[0], use_cache=False).logits, expected, atol=0, rtol=0
            )
