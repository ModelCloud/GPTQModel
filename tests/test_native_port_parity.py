# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 families must equal Transformers 5.23 exactly.

Each fixture under tests/fixtures/ports/native_parity was dumped from
Transformers' own model tester by scripts/generate_native_port_parity_fixtures.py.
For every model class, the native class (the oracle; tests may import
``transformers.models``) and the vendored class are built from the same config,
must have identical state_dict keys/shapes, and with identical weights must
produce bit-identical outputs.
"""

import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"
CASES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def _classes(family, module, name):
    native = getattr(importlib.import_module(f"transformers.models.{family}.{module}"), name)
    vendored = getattr(importlib.import_module(f"gptqmodel.models.ports.native.{family}.{module}"), name)
    return native, vendored


def _inputs(family, spec):
    tensors = {}
    path = FIXTURES / f"{family}.safetensors"
    if path.exists():
        tensors.update(load_file(str(path)))
    generator = torch.Generator().manual_seed(0)
    for key in sorted(spec["float_inputs"]):
        meta = spec["float_inputs"][key]
        dtype = getattr(torch, meta["dtype"])
        tensors[key] = torch.randn(meta["shape"], generator=generator, dtype=torch.float32).to(dtype)
    tensors.update(spec["extra_inputs"])
    return tensors


def _flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict) or hasattr(output, "to_tuple"):
        values = output.values() if isinstance(output, dict) else output.to_tuple()
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        return []
    return [tensor for value in values for tensor in _flatten(value)]


# Families whose *native* Transformers forward cannot run on the tester inputs here.
ORACLE_UNAVAILABLE = {
    "inkling": "native Inkling requires scipy",
    "roberta": "vendored only as a dependency; the tester's inputs target encoder heads",
}


@pytest.mark.parametrize("family", CASES)
def test_vendored_family_matches_transformers(family):
    if family in ORACLE_UNAVAILABLE:
        if family == "inkling":
            pytest.importorskip("scipy", reason=ORACLE_UNAVAILABLE[family])
        else:
            pytest.skip(ORACLE_UNAVAILABLE[family])
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    native_config_class, vendored_config_class = _classes(family, spec["config_module"], spec["config_class"])
    assert vendored_config_class is not native_config_class
    inputs = _inputs(family, spec)

    for module, name in spec["model_classes"]:
        native_class, vendored_class = _classes(family, module, name)
        native_config = native_config_class.from_dict(dict(spec["config"]))
        vendored_config = vendored_config_class.from_dict(dict(spec["config"]))
        torch.manual_seed(0)
        native = native_class(native_config).eval()
        torch.manual_seed(0)
        vendored = vendored_class(vendored_config).eval()

        native_state = native.state_dict()
        vendored_state = vendored.state_dict()
        assert list(vendored_state) == list(native_state), f"{name}: state_dict keys differ"
        assert all(vendored_state[k].shape == native_state[k].shape for k in native_state), f"{name}: shapes differ"
        vendored.load_state_dict(native_state, strict=True)
        for key, buffer in native.named_buffers():
            vendored.get_buffer(key).copy_(buffer)

        with torch.no_grad():
            expected = _flatten(native(**inputs))
            actual = _flatten(vendored(**inputs))
        assert expected, f"{name}: no tensor outputs"
        assert len(actual) == len(expected), f"{name}: output structure differs"
        for index, (want, got) in enumerate(zip(expected, actual)):
            assert torch.equal(want, got), f"{name}: output {index} differs"


def test_every_vendored_family_has_parity_coverage():
    native = Path(__file__).resolve().parents[1] / "gptqmodel" / "models" / "ports" / "native"
    families = {p.name for p in native.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"}}
    # No Transformers 5.23 model tester exists for these; they need hand-written parity tests.
    without_tester = {"bart", "llama4", "opt"}
    assert families - without_tester <= set(CASES), sorted(families - without_tester - set(CASES))


# Llama4 has no Transformers 5.23 model tester, so its parity check is hand-written: tiny text-only and
# conditional-generation (vision + MoE text) configs, default dict outputs plus every optional output path.
LLAMA4_TEXT = {
    "vocab_size": 99, "hidden_size": 32, "intermediate_size": 37, "intermediate_size_mlp": 64,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "num_local_experts": 4, "num_experts_per_tok": 1, "interleave_moe_layer_step": 2, "attention_chunk_size": 4,
    "max_position_embeddings": 64, "pad_token_id": 0, "no_rope_layer_interval": 2, "floor_scale": 4,
}
LLAMA4_VISION = {
    "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
    "vision_output_dim": 32, "image_size": 16, "patch_size": 4, "projector_input_dim": 32,
    "projector_output_dim": 32, "pixel_shuffle_ratio": 0.5,
}
LLAMA4_IMAGE_TOKEN = 98
LLAMA4_CALL_VARIANTS = (
    {},
    {"output_hidden_states": True, "output_attentions": True},
    {"return_dict": False},
)


def _llama4_config(config_module, name):
    if name == "Llama4TextConfig":
        return config_module.Llama4TextConfig(**LLAMA4_TEXT, attn_implementation="eager")
    return config_module.Llama4Config(
        text_config=dict(LLAMA4_TEXT), vision_config=dict(LLAMA4_VISION),
        image_token_index=LLAMA4_IMAGE_TOKEN, attn_implementation="eager",
    )


def _llama4_inputs(model_name):
    pixel_values = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    if model_name == "Llama4VisionModel":
        return {"pixel_values": pixel_values}
    input_ids = torch.randint(1, 90, (2, 9), generator=torch.Generator().manual_seed(0))
    if model_name != "Llama4ForConditionalGeneration":
        return {"input_ids": input_ids}
    input_ids[:, 1:5] = LLAMA4_IMAGE_TOKEN  # 4 tokens per 16x16 image after pixel shuffle
    return {"input_ids": input_ids, "pixel_values": pixel_values}


@pytest.mark.parametrize("model_name", [
    "Llama4TextModel", "Llama4ForCausalLM", "Llama4VisionModel", "Llama4ForConditionalGeneration",
])
def test_vendored_llama4_matches_transformers(model_name):
    native_config_module = importlib.import_module("transformers.models.llama4.configuration_llama4")
    vendored_config_module = importlib.import_module("gptqmodel.models.ports.native.llama4.configuration_llama4")
    config_name = "Llama4TextConfig" if model_name in {"Llama4TextModel", "Llama4ForCausalLM"} else "Llama4Config"
    native_config = _llama4_config(native_config_module, config_name)
    vendored_config = _llama4_config(vendored_config_module, config_name)
    assert type(vendored_config) is not type(native_config)
    if model_name == "Llama4VisionModel":
        native_config, vendored_config = native_config.vision_config, vendored_config.vision_config

    native_class, vendored_class = _classes("llama4", "modeling_llama4", model_name)
    torch.manual_seed(0)
    native = native_class(native_config).eval()
    torch.manual_seed(0)
    vendored = vendored_class(vendored_config).eval()
    native_state = native.state_dict()
    assert list(vendored.state_dict()) == list(native_state)
    vendored.load_state_dict(native_state, strict=True)
    for key, buffer in native.named_buffers():
        vendored.get_buffer(key).copy_(buffer)

    inputs = _llama4_inputs(model_name)
    for variant in LLAMA4_CALL_VARIANTS:
        with torch.no_grad():
            expected_output = native(**inputs, **variant)
            actual_output = vendored(**inputs, **variant)
        assert type(actual_output).__name__ == type(expected_output).__name__, variant
        expected, actual = _flatten(expected_output), _flatten(actual_output)
        assert expected and len(actual) == len(expected), variant
        for index, (want, got) in enumerate(zip(expected, actual)):
            assert torch.equal(want, got), f"{model_name} {variant}: output {index} differs"
