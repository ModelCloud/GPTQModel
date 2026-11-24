# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definitions wired to the vendored Transformers 5.23 ports load only vendored classes.

Each case validates a tiny config with the *vendored* config class, saves a
checkpoint with the native Transformers class, and loads it through the public
definition loader
(``MODEL_MAP[model_type].loader``). Every module must come from GPT-QModel and
none from ``transformers.models``; outputs must equal the native Transformers
class (the oracle) loaded from the same checkpoint.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import transformers
from eager_ssm_kernels import eager_torch_ssm_kernels
from safetensors.torch import load_file

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.native.auto import CONFIG_MAPPING


@pytest.fixture(autouse=True)
def _pin_eager_torch_ssm_kernels():
    """Run this module's CPU forwards on the Torch SSM/DeltaNet reference.

    ``transformers`` binds ``causal_conv1d``/``mamba_ssm``/``fla`` at import time
    when the distribution is installed, and those implementations are CUDA-only.
    The invariants here feed CPU tensors, so pin the reference kept on
    ``__wrapped__`` (upstream precedent #5537) instead of failing on the device.
    """

    with eager_torch_ssm_kernels():
        yield

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"
NATIVE_PREFIX = "gptqmodel.models.ports.native."

ORACLE = {
    "NativeForCausalLM": "AutoModelForCausalLM",
    "NativeForImageTextToText": "AutoModelForImageTextToText",
    "NativeModel": "AutoModel",
}


def _fixture(family):
    return json.loads((FIXTURES / f"{family}.json").read_text())


def _fixture_config(family, sub=None):
    config = copy.deepcopy(_fixture(family)["config"])
    return config[sub] if sub else config


def _gemma3n_text_config():
    return {
        "model_type": "gemma3n_text",
        "vocab_size": 128,
        "vocab_size_per_layer_input": 128,
        "hidden_size": 32,
        "hidden_size_per_layer_input": 8,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        "num_kv_shared_layers": 2,
        "laurel_rank": 4,
        "altup_num_inputs": 2,
        "activation_sparsity_pattern": [0.0, 0.0, 0.0, 0.0],
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _gemma3n_config():
    text = _gemma3n_text_config()
    return {
        "model_type": "gemma3n",
        "text_config": text,
        "audio_config": _fixture_config("gemma3n"),
        "vision_config": {
            "model_type": "gemma3n_vision",
            "architecture": "mobilenetv5_300m_enc",
            "hidden_size": 2048,
            "vocab_size": 128,
            "vocab_offset": 128 + 128,
        },
        "boi_token_id": 120,
        "eoi_token_id": 121,
        "image_token_id": 122,
        "boa_token_id": 123,
        "eoa_token_id": 124,
        "audio_token_id": 125,
    }


# model_type -> tiny config dict (vendored config class is selected by model_type).
CASES = {
    "bloom": lambda: _fixture_config("bloom"),
    "decision_transformer": lambda: _fixture_config("decision_transformer"),
    "deepseek_vl": lambda: _fixture_config("deepseek_vl"),
    "ernie4_5": lambda: _fixture_config("ernie4_5"),
    "falcon_mamba": lambda: _fixture_config("falcon_mamba"),
    "gemma3n": _gemma3n_config,
    "gemma3n_text": _gemma3n_text_config,
    "glm4_moe_lite": lambda: _fixture_config("glm4_moe_lite"),
    "glm4v_moe": lambda: _fixture_config("glm4v_moe"),
    "glm4v_moe_text": lambda: _fixture_config("glm4v_moe", "text_config"),
    "glm_ocr": lambda: _fixture_config("glm_ocr"),
    "gpt_neo": lambda: _fixture_config("gpt_neo"),
    "granite_swa": lambda: _fixture_config("granite_swa"),
    "hrm_text": lambda: _fixture_config("hrm_text"),
    "hy_v3": lambda: _fixture_config("hy_v3"),
    "laguna": lambda: _fixture_config("laguna"),
    "lfm3": lambda: _fixture_config("lfm3"),
    "minicpm3": lambda: _fixture_config("minicpm3"),
    "ministral3": lambda: _fixture_config("ministral3"),
    "muse_glimmer": lambda: _fixture_config("muse_glimmer"),
    "olmoe": lambda: _fixture_config("olmoe"),
    "qwen2": lambda: _fixture_config("qwen2"),
    "qwen3_5": lambda: _fixture_config("qwen3_5"),
    "qwen3_moe": lambda: _fixture_config("qwen3_moe"),
    # Inherits Qwen3MoeQModel's loader.
    "qwen3_5_moe_text": lambda: _fixture_config("qwen3_5_moe", "text_config"),
    "qwen3_vl": lambda: _fixture_config("qwen3_vl"),
    "qwen3_vl_moe": lambda: _fixture_config("qwen3_vl_moe"),
    "solar_open": lambda: _fixture_config("solar_open"),
    "zamba2": lambda: _fixture_config("zamba2"),
}

TIMM_REQUIRED = {"gemma3n"}


def _special_token_ids(config):
    ids = set()
    for key, value in config.items():
        if isinstance(value, dict):
            ids |= _special_token_ids(value)
        elif key.endswith("token_id") and isinstance(value, int):
            ids.add(value)
    return ids


def _vocab_size(config):
    if "vocab_size" in config and "text_config" not in config:
        return config["vocab_size"]
    return config["text_config"]["vocab_size"]


def _inputs(model_type, config):
    if model_type == "decision_transformer":
        spec = _fixture("decision_transformer")
        inputs = dict(load_file(str(FIXTURES / "decision_transformer.safetensors")))
        generator = torch.Generator().manual_seed(0)
        for key in sorted(spec["float_inputs"]):
            meta = spec["float_inputs"][key]
            inputs[key] = torch.randn(meta["shape"], generator=generator).to(getattr(torch, meta["dtype"]))
        return inputs
    special = _special_token_ids(config)
    candidates = [token for token in range(3, _vocab_size(config)) if token not in special]
    generator = torch.Generator().manual_seed(0)
    picks = torch.randint(len(candidates), (2, 7), generator=generator)
    return {"input_ids": torch.tensor(candidates)[picks]}


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


def _assert_vendored(model):
    leaked = sorted({type(m).__module__ for m in model.modules() if type(m).__module__.startswith("transformers.models")})
    assert not leaked, f"modules from transformers.models: {leaked}"
    assert type(model).__module__.startswith(NATIVE_PREFIX), type(model).__module__
    assert type(model.config).__module__.startswith(NATIVE_PREFIX), type(model.config).__module__
    for sub in ("text_config", "vision_config", "audio_config"):
        sub_config = getattr(model.config, sub, None)
        if sub_config is not None:
            assert type(sub_config).__module__.startswith(NATIVE_PREFIX), type(sub_config).__module__


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_definition_loader_builds_vendored_classes(model_type, tmp_path):
    if model_type in TIMM_REQUIRED:
        pytest.importorskip("timm")
    loader = MODEL_MAP[model_type].loader
    assert isinstance(loader, LocalModelLoader)
    head = loader.model_class.rpartition(".")[2]
    assert loader.config_class == f"{NATIVE_PREFIX}auto.NativeConfig"
    assert head in ORACLE

    config_dict = CASES[model_type]()
    config_dict["model_type"] = model_type
    vendored_config_class = CONFIG_MAPPING[model_type]
    assert vendored_config_class.__module__.startswith(NATIVE_PREFIX)
    vendored_config_class.from_dict(copy.deepcopy(config_dict))
    # The checkpoint is written by native Transformers, so loading it through the
    # definition exercises Transformers' checkpoint conversion mappings.
    native_config = transformers.AutoConfig.for_model(**copy.deepcopy(config_dict))
    assert type(native_config).__module__.startswith("transformers.models.")
    torch.manual_seed(0)
    native = getattr(transformers, ORACLE[head]).from_config(native_config)
    native.save_pretrained(tmp_path)
    del native

    loaded_config = loader.load_config(str(tmp_path))
    assert type(loaded_config) is vendored_config_class
    model = loader.from_pretrained(str(tmp_path), config=loaded_config).eval()
    _assert_vendored(model)

    oracle = getattr(transformers, ORACLE[head]).from_pretrained(str(tmp_path)).eval()
    assert type(oracle).__module__.startswith("transformers.models.")
    assert type(oracle).__name__ == type(model).__name__

    inputs = _inputs(model_type, config_dict)
    with torch.no_grad():
        expected = _flatten(oracle(**inputs))
        actual = _flatten(model(**inputs))
    assert expected
    assert len(actual) == len(expected)
    # A GPT-QModel load earlier in the session may have let Defuser swap native
    # Transformers MoE blocks for per-expert ones (the vendored classes are never
    # patched); that oracle only differs by summation order.
    defused_oracle = any(type(m).__module__.startswith("defuser.") for m in oracle.modules())
    for want, got in zip(expected, actual):
        if defused_oracle:
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(want, got)


@pytest.mark.parametrize(
    "model_type",
    ["deepseek_vl", "gemma3n", "muse_glimmer", "qwen3_5", "qwen3_vl", "qwen3_vl_moe"],
)
def test_processor_definitions_declare_vendored_processor(model_type):
    loader = MODEL_MAP[model_type].loader
    assert MODEL_MAP[model_type].require_load_processor
    assert loader.processor_class == f"{NATIVE_PREFIX}auto.NativeProcessor"
    assert loader.tokenizer_class == f"{NATIVE_PREFIX}auto.NativeTokenizer"


def test_qwen3_5_subclasses_keep_their_own_loaders():
    for model_type in ("sensenova_loop", "qwen_drive", "zdtaichu5_0"):
        definition = MODEL_MAP[model_type]
        assert isinstance(definition.loader, LocalModelLoader)
        assert definition.loader is not MODEL_MAP["qwen3_5"].loader
