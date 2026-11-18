# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Group-0 definitions load vendored Transformers 5.23 classes through their public loader.

Each case builds a tiny checkpoint from the Transformers model-tester config
recorded in tests/fixtures/ports/native_parity, saves it, and loads it through
``MODEL_MAP[model_type].loader``. Every instantiated module must come from
GPT-QModel (or torch), never from ``transformers.models``, and the logits must
equal the native Transformers class (the oracle) loaded from the same files.
"""

import json
from pathlib import Path

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.native.auto import CONFIG_MAPPING


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"


def _fixture_config(family):
    return dict(json.loads((FIXTURES / f"{family}.json").read_text())["config"])


def _gemma3_config():
    return {
        "model_type": "gemma3",
        "text_config": _fixture_config("gemma3"),
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
        },
        "mm_tokens_per_image": 4,
    }


def _qwen3_5_text_config():
    config = dict(_fixture_config("qwen3_5")["text_config"])
    config["model_type"] = "qwen3_5_text"
    return config


# model_type -> (config dict factory, native Transformers auto class name)
CASES = {
    "axk2": (lambda: _fixture_config("axk2"), "AutoModelForCausalLM"),
    "dbrx": (lambda: _fixture_config("dbrx"), "AutoModelForCausalLM"),
    "deepseek_v4": (lambda: _fixture_config("deepseek_v4"), "AutoModelForCausalLM"),
    "dots3_note": (lambda: _fixture_config("dots3_note"), "AutoModelForCausalLM"),
    "falcon_h1": (lambda: _fixture_config("falcon_h1"), "AutoModelForCausalLM"),
    "gemma3": (_gemma3_config, "AutoModelForCausalLM"),
    "gemma3_text": (lambda: _fixture_config("gemma3"), "AutoModelForCausalLM"),
    "glm": (lambda: _fixture_config("glm"), "AutoModelForCausalLM"),
    "glm4": (lambda: _fixture_config("glm4"), "AutoModelForCausalLM"),
    "glm4_moe": (lambda: _fixture_config("glm4_moe"), "AutoModelForCausalLM"),
    "glm_moe_dsa": (lambda: _fixture_config("glm_moe_dsa"), "AutoModelForCausalLM"),
    "gpt_bigcode": (lambda: _fixture_config("gpt_bigcode"), "AutoModelForCausalLM"),
    "gptj": (lambda: _fixture_config("gptj"), "AutoModelForCausalLM"),
    "granitemoehybrid": (lambda: _fixture_config("granitemoehybrid"), "AutoModelForCausalLM"),
    "hunyuan_vl": (lambda: _fixture_config("hunyuan_vl"), "AutoModelForImageTextToText"),
    "jamba": (lambda: _fixture_config("jamba"), "AutoModelForCausalLM"),
    "lfm2_vl": (lambda: _fixture_config("lfm2_vl"), "AutoModelForImageTextToText"),
    "mimo_v2_flash": (lambda: _fixture_config("mimo_v2_flash"), "AutoModelForCausalLM"),
    "minimax_m3_vl": (lambda: _fixture_config("minimax_m3_vl"), "AutoModelForImageTextToText"),
    "mpt": (lambda: _fixture_config("mpt"), "AutoModelForCausalLM"),
    "olmo_hybrid": (lambda: _fixture_config("olmo_hybrid"), "AutoModelForCausalLM"),
    "phi": (lambda: _fixture_config("phi"), "AutoModelForCausalLM"),
    "phi3": (lambda: _fixture_config("phi3"), "AutoModelForCausalLM"),
    "phimoe": (lambda: _fixture_config("phimoe"), "AutoModelForCausalLM"),
    "qwen2_moe": (lambda: _fixture_config("qwen2_moe"), "AutoModelForCausalLM"),
    "qwen3_5_text": (_qwen3_5_text_config, "AutoModelForCausalLM"),
    "smolvlm": (lambda: _fixture_config("smolvlm"), "AutoModelForImageTextToText"),
    "zamba": (lambda: _fixture_config("zamba"), "AutoModelForCausalLM"),
}

PROCESSOR_MODEL_TYPES = {"gemma3", "hunyuan_vl", "lfm2_vl", "smolvlm"}


def _special_token_ids(config):
    ids = set()
    for cfg in (config, config.get_text_config()):
        for key, value in cfg.to_dict().items():
            if (key.endswith("_token_id") or key.endswith("_token_index")) and isinstance(value, int):
                ids.add(value)
    return ids


def _input_ids(config):
    vocab = int(config.get_text_config().vocab_size)
    special = _special_token_ids(config)
    allowed = torch.tensor([i for i in range(1, vocab) if i not in special])
    generator = torch.Generator().manual_seed(0)
    return allowed[torch.randint(len(allowed), (1, 7), generator=generator)]


def _reset_derived_buffers(model_type, payload, native_auto, *models):
    """Recompute non-persistent buffers from the config for every loaded model.

    ``from_pretrained`` leaves some config-derived, non-persistent buffers (for
    example Falcon-H1's MuP vector) uninitialized in Transformers 5.23 itself;
    the definitions rebuild them after load. Take them from a native model
    constructed directly from the config so the comparison is deterministic.
    """
    import transformers

    config = transformers.CONFIG_MAPPING[model_type].from_dict(payload)
    config._attn_implementation = "eager"
    reference = getattr(transformers, native_auto).from_config(config)
    persistent = set(reference.state_dict())
    for name, buffer in reference.named_buffers():
        if name in persistent:
            continue
        for model in models:
            model.get_buffer(name).copy_(buffer)


def _loader(model_type):
    loader = MODEL_MAP[model_type].loader
    assert isinstance(loader, LocalModelLoader)
    return loader


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_definition_loader_builds_vendored_classes(model_type, tmp_path):
    import transformers

    factory, native_auto = CASES[model_type]
    loader = _loader(model_type)
    payload = factory()
    config_class = CONFIG_MAPPING[model_type]
    assert config_class.__module__.startswith("gptqmodel.models.ports.native.")
    config = config_class.from_dict(payload)
    config._attn_implementation = "eager"

    torch.manual_seed(0)
    built = loader.from_config(config).eval()
    built.save_pretrained(tmp_path)
    del built

    loaded_config = loader.load_config(tmp_path)
    # dots3_note's causal-LM shell is the text-only model, which saves its text config.
    assert type(loaded_config) is CONFIG_MAPPING[loaded_config.model_type]
    assert type(loaded_config).__module__.startswith("gptqmodel.models.ports.native.")
    model = loader.from_pretrained(tmp_path, config=loaded_config, attn_implementation="eager").eval()
    assert type(model).__module__.startswith("gptqmodel.models.ports.native.")
    foreign = sorted({
        f"{type(module).__module__}.{type(module).__name__}"
        for module in model.modules()
        if type(module).__module__.startswith("transformers.models")
    })
    assert not foreign, foreign

    # Oracle: the native Transformers class loaded from the same checkpoint.
    native = getattr(transformers, native_auto).from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert type(native).__module__.startswith("transformers.models.")
    assert type(native).__name__ == type(model).__name__
    _reset_derived_buffers(model_type, payload, native_auto, model, native)

    input_ids = _input_ids(loaded_config)
    with torch.no_grad():
        expected = native(input_ids=input_ids).logits
        actual = model(input_ids=input_ids).logits
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("model_type", sorted(PROCESSOR_MODEL_TYPES))
def test_processor_definitions_declare_vendored_processor(model_type):
    from gptqmodel.models.local import _local_class
    from gptqmodel.models.ports.native.auto import _registry

    loader = _loader(model_type)
    assert loader.processor_class is not None
    assert MODEL_MAP[model_type].require_load_processor
    assert _local_class(loader.processor_class) is not None
    assert model_type in _registry.PROCESSOR
