# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real native Qwen2.5-VL decoder and explicit text-head qualification."""

import copy
import json

import pytest
import torch
from transformers import PreTrainedTokenizerFast
from transformers import Qwen2_5_VLConfig as OracleConfig
from transformers import Qwen2_5_VLForConditionalGeneration as OracleComposite
from transformers import Qwen2_5_VLTextConfig as OracleTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel as OracleTextModel

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.qwen2_5_vl import Qwen2_5_VLQModel
from gptqmodel.models.ports.native.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from gptqmodel.models.ports.native.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
)


def oracle(config):
    """Transformers 5.23's own config class: the numerical oracle for the vendored port."""
    oracle_class = {"qwen2_5_vl": OracleConfig, "qwen2_5_vl_text": OracleTextConfig}[config.model_type]
    return oracle_class.from_dict(copy.deepcopy(config).to_dict())


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


def test_text_alias_is_exact_native_headless_backbone_and_root_tree():
    config = text_config()
    model = Qwen2_5_VLQModel.loader.from_config(config, attn_implementation="eager").eval()
    ids = torch.tensor([[1, 4, 9, 3]])
    with torch.no_grad():
        output = model(ids, use_cache=True)
    assert type(model) is Qwen2_5_VLTextModel
    assert output.last_hidden_state.shape == (1, 4, 32)
    assert not hasattr(model, "lm_head") and model.get_output_embeddings() is None
    assert not model.can_generate() and not hasattr(model, "generate")
    assert output.past_key_values.get_seq_length() == 4
    tree = Qwen2_5_VLQModel._resolve_effective_module_tree(model, None)
    with Qwen2_5_VLQModel._module_tree_context(tree):
        assert Qwen2_5_VLQModel.extract_layers_node() == ["layers"]


def test_composite_remains_exact_native_class():
    model = Qwen2_5_VLQModel.loader.from_config(composite_config(), attn_implementation="eager")
    assert type(model) is Qwen2_5_VLForConditionalGeneration


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_text_exact_native_state_schema_and_hidden_output(kv_heads):
    model = Qwen2_5_VLQModel.loader.from_config(text_config(kv_heads), attn_implementation="eager").eval()
    ids = torch.tensor([[1, 4, 9, 3], [1, 8, 5, 7]])
    with torch.no_grad():
        native = OracleTextModel._from_config(oracle(model.config), attn_implementation="eager").eval()
        native.load_state_dict(model.state_dict())
        assert set(model.state_dict()) == set(native.state_dict())
        assert not any(key.startswith("model.") or "lm_head" in key for key in model.state_dict())
        expected = native(ids).last_hidden_state
        actual = model(ids).last_hidden_state
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


@pytest.mark.parametrize("dtype,implementation,sliding", [
    (torch.float32, "eager", False), (torch.bfloat16, "sdpa", True), (torch.float16, "eager", True),
])
def test_text_matches_native_composite_decoder_hidden_states_and_cache(dtype, implementation, sliding):
    config = composite_config()
    config.text_config.use_sliding_window = sliding
    config.text_config.sliding_window = 3 if sliding else None
    config.text_config.layer_types = ["full_attention", "sliding_attention"] if sliding else ["full_attention"] * 2
    text = Qwen2_5_VLQModel.loader.from_config(
        copy.deepcopy(config.text_config),
        dtype=dtype,
        attn_implementation=implementation,
    ).eval()
    native = OracleComposite._from_config(
        oracle(config), dtype=dtype, attn_implementation=implementation
    ).eval()
    native = native.model.language_model
    native.load_state_dict(text.state_dict())
    ids = torch.tensor([[0, 1, 5, 8, 7], [1, 6, 4, 9, 3]])
    mask = (ids != 0).long()
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    with torch.inference_mode():
        expected = native(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        actual = text(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=0, rtol=0)
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
        torch.testing.assert_close(actual_next.last_hidden_state, expected_next.last_hidden_state, atol=0, rtol=0)
        assert actual_next.past_key_values.get_seq_length() == 7
        assert not text.can_generate()


@pytest.mark.parametrize("dtype,tied", [(torch.float32, False), (torch.bfloat16, True)])
def test_text_safetensors_two_cycles_assignment_and_failure_repair(tmp_path, dtype, tied):
    model = make_text_model(dtype=dtype, tie_word_embeddings=tied)
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        expected = model(ids).last_hidden_state
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for assign in (False, True):
        target = make_text_model(dtype=dtype, tie_word_embeddings=tied)
        target.load_state_dict(snapshot, assign=assign)
        with torch.inference_mode():
            torch.testing.assert_close(target(ids).last_hidden_state, expected, rtol=0, atol=0)
    for cycle in range(2):
        path = tmp_path / str(cycle)
        model.save_pretrained(path, safe_serialization=True)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        from gptqmodel.utils.hf import load_model_config

        config = load_model_config(path, trust_remote_code=True)
        model = Qwen2_5_VLQModel.loader.from_pretrained(
            path, config=config, dtype=dtype, attn_implementation="eager"
        ).eval()
        assert type(model) is Qwen2_5_VLTextModel
        assert not hasattr(model, "lm_head") and model.get_output_embeddings() is None
        assert model.state_dict().keys() == snapshot.keys()
        for key, value in model.state_dict().items():
            assert value.dtype == snapshot[key].dtype
            torch.testing.assert_close(value, snapshot[key], atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).last_hidden_state, expected, atol=0, rtol=0)
    broken_path = tmp_path / "1" / "config.json"
    original = broken_path.read_bytes()
    broken_path.write_text('{"model_type":"foreign"}')
    with pytest.raises(ValueError, match="Unsupported Qwen2.5-VL"):
        Qwen2_5_VLQModel.loader.load_config(tmp_path / "1")
    broken_path.write_bytes(original)
    repaired = Qwen2_5_VLQModel.loader.from_pretrained(tmp_path / "1", dtype=dtype, attn_implementation="eager").eval()
    with torch.inference_mode():
        torch.testing.assert_close(repaired(ids).last_hidden_state, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bypass,trust,text_only", [
    (None, None, False), ("1", True, False), ("0", False, True), ("1", None, True),
])
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
        else OracleComposite._from_config(
            oracle(composite_config()),
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
    tokenizer_path = tmp_path / "tokenizer_config.json"
    tokenizer_payload = json.loads(tokenizer_path.read_text())
    tokenizer_payload["auto_map"] = {"AutoTokenizer": ["evil.Evil", "evil.Evil"]}
    tokenizer_path.write_text(json.dumps(tokenizer_payload))
    loaded = GPTQModel.load(
        str(tmp_path), backend=BACKEND.TORCH, dtype=torch.float32, trust_remote_code=trust, attn_implementation="eager"
    )
    assert loaded.trust_remote_code is False
    assert type(loaded.model) is (Qwen2_5_VLTextModel if text_only else Qwen2_5_VLForConditionalGeneration)
    assert loaded.require_load_processor is (not text_only)
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        if text_only:
            assert type(loaded.model) is Qwen2_5_VLTextModel
            assert not hasattr(loaded.model, "lm_head")
            torch.testing.assert_close(
                loaded.model(ids).last_hidden_state, model(ids).last_hidden_state, atol=0, rtol=0
            )
            with pytest.raises(NotImplementedError, match="headless backbone"):
                loaded.generate(input_ids=ids, max_new_tokens=2)
        else:
            torch.testing.assert_close(loaded.model(ids).logits, model(ids).logits, atol=0, rtol=0)
            generation = {"max_new_tokens": 2, "do_sample": False, "eos_token_id": None}
            torch.testing.assert_close(
                loaded.generate(input_ids=ids, **generation),
                model.generate(input_ids=ids, **generation),
                atol=0,
                rtol=0,
            )
    assert not marker.exists()


def test_text_native_outputs_and_input_embedding_equivalence():
    model = make_text_model()
    ids = torch.tensor([[1, 4, 7, 5]])
    with torch.inference_mode():
        output = model(ids, output_hidden_states=True, output_attentions=True)
        assert not hasattr(output, "logits") and not hasattr(output, "loss")
        torch.testing.assert_close(model(ids, return_dict=False)[0], output.last_hidden_state)
        torch.testing.assert_close(
            model(inputs_embeds=model.get_input_embeddings()(ids)).last_hidden_state, output.last_hidden_state
        )
        assert len(output.hidden_states) == 3 and len(output.attentions) == 2


@pytest.mark.parametrize("beams", [1, 3])
def test_native_composite_media_forward_and_generation_unchanged(beams):
    config = composite_config()
    model = Qwen2_5_VLQModel.loader.from_config(config, attn_implementation="eager").eval()
    native = OracleComposite._from_config(oracle(config), attn_implementation="eager").eval()
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


def test_public_headless_dense_two_save_cycles_and_generation_rejection(tmp_path):
    from gptqmodel import BACKEND, GPTQModel

    model = make_text_model()
    source = tmp_path / "source"
    model.save_pretrained(source)
    save_tokenizer(source)
    ids = torch.tensor([[1, 4, 6, 5]])
    with torch.inference_mode():
        expected = model(ids).last_hidden_state
    state = {key: value.clone() for key, value in model.state_dict().items()}
    loaded = GPTQModel.load(str(source), backend=BACKEND.TORCH, dtype=torch.float32, attn_implementation="eager")
    for index in range(2):
        destination = tmp_path / str(index)
        with pytest.raises(NotImplementedError, match="headless backbone"):
            loaded.generate(input_ids=ids, max_new_tokens=1)
        loaded.model.save_pretrained(destination, safe_serialization=True)
        loaded.tokenizer.save_pretrained(str(destination))
        assert list(destination.glob("*.safetensors")) and not list(destination.glob("*.bin"))
        loaded = GPTQModel.load(
            str(destination), backend=BACKEND.TORCH, dtype=torch.float32, attn_implementation="eager"
        )
        assert type(loaded.model) is Qwen2_5_VLTextModel
        assert loaded.model.state_dict().keys() == state.keys()
        for key, value in state.items():
            torch.testing.assert_close(loaded.model.state_dict()[key], value, atol=0, rtol=0)
        with torch.inference_mode():
            torch.testing.assert_close(loaded.model(ids).last_hidden_state, expected, atol=0, rtol=0)


@pytest.mark.parametrize("offload,kv_heads", [(False, 1), (True, 4)])
def test_public_awq_packs_all_linears_and_reloads_twice(tmp_path, offload, kv_heads):
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
    assert wrapped.extract_layers_node() == ["layers"]
    wrapped.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    packed = {name for name, module in wrapped.model.named_modules() if isinstance(module, AwqTorchLinear)}
    assert packed == {
        f"layers.{index}.{projection}"
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
        actual = wrapped.model.state_dict()[key]
        expect_fold = key.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight"))
        assert torch.equal(actual, value) is not expect_fold, key
    with torch.inference_mode():
        expected = wrapped.model(**calibration[0], use_cache=False).last_hidden_state
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
                wrapped.model(**calibration[0], use_cache=False).last_hidden_state, expected, atol=0, rtol=0
            )
