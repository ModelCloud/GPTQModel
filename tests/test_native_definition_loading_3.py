# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Group-3 definitions load only vendored Transformers 5.23 classes.

Each case builds a tiny checkpoint with the *vendored* config/model classes,
loads it through the public ``MODEL_MAP[model_type].loader`` path and checks
that no module comes from ``transformers.models``. The native Transformers
class (the oracle; tests may import ``transformers.models``) loads the same
checkpoint and must produce identical logits.
"""

import importlib
import json
from pathlib import Path

import pytest
import torch
from eager_ssm_kernels import eager_torch_ssm_kernels
from safetensors.torch import load_file

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader, _local_class


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
NATIVE = "gptqmodel.models.ports.native."


def _fixture(family):
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    tensors = load_file(str(FIXTURES / f"{family}.safetensors"))
    generator = torch.Generator().manual_seed(0)
    for key in sorted(spec["float_inputs"]):
        meta = spec["float_inputs"][key]
        tensors[key] = torch.randn(meta["shape"], generator=generator).to(getattr(torch, meta["dtype"]))
    tensors.update(spec["extra_inputs"])
    return spec, tensors


def _vendored(family, module, name):
    return getattr(importlib.import_module(f"{NATIVE}{family}.{module}"), name)


def _fixture_case(family, *, sub_config=None, config_class=None, text_only=False):
    def build():
        spec, inputs = _fixture(family)
        payload = dict(spec["config"])
        if sub_config is not None:
            payload = dict(payload[sub_config])
        config = _vendored(family, spec["config_module"], config_class or spec["config_class"]).from_dict(payload)
        if text_only:
            inputs = {key: inputs[key] for key in ("input_ids", "attention_mask") if key in inputs}
        return config, inputs

    return build


def _mllama_case():
    spec, inputs = _fixture("mllama")
    config_class = _vendored("mllama", "configuration_mllama", "MllamaConfig")
    text = dict(spec["config"], cross_attention_layers=[1])
    vision = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_global_layers": 1,
        "attention_heads": 4, "image_size": 8, "patch_size": 4, "max_num_tiles": 4,
        "intermediate_layers_indices": [0], "vision_output_dim": 32,
    }
    config = config_class(text_config=text, vision_config=vision)
    return config, {key: inputs[key] for key in ("input_ids", "attention_mask")}


def _qwen3_omni_case():
    spec, inputs = _fixture("qwen3_omni_moe")
    config_class = _vendored("qwen3_omni_moe", "configuration_qwen3_omni_moe", "Qwen3OmniMoeConfig")
    config = config_class(thinker_config=dict(spec["config"]), enable_audio_output=False)
    # Text tokens only (ids 0..8 are audio/vision specials); tower parity is in test_native_port_parity.
    input_ids = torch.randint(9, config.thinker_config.text_config.vocab_size, inputs["input_ids"].shape,
                              generator=torch.Generator().manual_seed(0))
    return config, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _glm5_next_case():
    # The integrated MTP loader requires the checkpoint's 45 target layers.
    config_class = _vendored("glm5_next", "configuration_glm5_next", "Glm5NextConfig")
    text = {
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "moe_intermediate_size": 16, "num_hidden_layers": 45,
        "num_attention_heads": 4, "num_key_value_heads": 4, "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 1, "q_lora_rank": 16, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 0,
        "v_head_dim": 8, "index_topk": 4, "index_kpool": 2, "index_head_dim": 8, "index_n_heads": 2, "linear_head_dim": 8,
        "linear_num_heads": 4, "hc_mult": 2, "max_position_embeddings": 32, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
    }
    vision = {
        "depth": 1, "hidden_size": 16, "intermediate_size": 32, "projection_intermediate_size": 32,
        "out_hidden_size": 32, "num_heads": 4, "in_channels": 3, "image_size": 4, "patch_size": 2,
        "spatial_merge_size": 1, "temporal_patch_size": 1,
    }
    config = config_class(
        text_config=text, vision_config=vision, image_token_id=60, video_token_id=61, image_start_token_id=62,
        image_end_token_id=63, video_start_token_id=58, video_end_token_id=59,
    )
    input_ids = torch.randint(3, 58, (2, 9), generator=torch.Generator().manual_seed(0))
    return config, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


CASES = {
    "afmoe": _fixture_case("afmoe"),
    "apertus": _fixture_case("apertus"),
    "cohere_compass": _fixture_case("cohere_compass"),
    "deepseek_v32": _fixture_case("deepseek_v32"),
    "dots1": _fixture_case("dots1"),
    "exaone4_5": _fixture_case("exaone4_5"),
    "gemma2": _fixture_case("gemma2"),
    "gemma4_unified": _fixture_case("gemma4_unified"),
    "gemma4_unified_text": _fixture_case(
        "gemma4_unified", sub_config="text_config", config_class="Gemma4UnifiedTextConfig", text_only=True
    ),
    "glm5_next": _glm5_next_case,
    "gpt2": _fixture_case("gpt2"),
    "gpt_oss": _fixture_case("gpt_oss"),
    "granitemoe_swa": _fixture_case("granitemoe_swa"),
    "hunyuan_v1_moe": _fixture_case("hunyuan_v1_moe"),
    "internvl": _fixture_case("internvl"),
    "lfm2_moe": _fixture_case("lfm2_moe"),
    "longcat_flash": _fixture_case("longcat_flash"),
    "minimax_m2": _fixture_case("minimax_m2"),
    "mixtral": _fixture_case("mixtral"),
    "mllama": _mllama_case,
    "mllama_text_model": _fixture_case("mllama"),
    "olmo3": _fixture_case("olmo3"),
    "ovis2": _fixture_case("ovis2"),
    "qwen2_audio": _fixture_case("qwen2_audio"),
    "qwen3_5_moe_text": _fixture_case(
        "qwen3_5_moe", sub_config="text_config", config_class="Qwen3_5MoeTextConfig", text_only=True
    ),
    "qwen3_omni_moe": _qwen3_omni_case,
    "voxtral": _fixture_case("voxtral"),
}


def _target_class(loader, config):
    model_class = _local_class(loader.model_class)
    return model_class._model_class(config) if hasattr(model_class, "_model_class") else model_class


def _native(vendored_class):
    native_module = importlib.import_module(vendored_class.__module__.replace(NATIVE, "transformers.models."))
    return getattr(native_module, vendored_class.__name__)


def _logits(model, inputs):
    # Qwen3-Omni's TextToWaveform wrapper generates audio; the thinker owns the logits.
    runner = model.thinker if hasattr(model, "thinker") else model
    with torch.no_grad():
        return runner(**inputs).logits


def test_cases_cover_every_group_definition():
    assert set(CASES) <= set(MODEL_MAP)
    assert all(isinstance(MODEL_MAP[model_type].loader, LocalModelLoader) for model_type in CASES)


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_definition_loader_builds_vendored_classes(model_type, tmp_path):
    config, inputs = CASES[model_type]()
    assert type(config).__module__.startswith(NATIVE)
    assert config.model_type == model_type
    loader = MODEL_MAP[model_type].loader

    # The checkpoint is written by the *native* Transformers class so loading
    # exercises the checkpoint conversion mappings (key renames, expert fusion).
    torch.manual_seed(0)
    native_config = _native(type(config)).from_dict(config.to_dict())
    _native(_target_class(loader, config))(native_config).eval().save_pretrained(tmp_path)

    loaded_config = loader.load_config(str(tmp_path))
    assert type(loaded_config) is type(config)
    model = loader.from_pretrained(str(tmp_path), config=loaded_config).eval()

    assert type(model).__module__.startswith(NATIVE), type(model).__module__
    offenders = sorted({
        f"{type(module).__module__}.{type(module).__name__}"
        for module in model.modules()
        if type(module).__module__.startswith("transformers.models")
        or (
            type(module).__module__.startswith("gptqmodel.")
            and not type(module).__module__.startswith((NATIVE, MODEL_MAP[model_type].__module__))
        )
    })
    assert not offenders, offenders

    oracle = _native(type(model)).from_pretrained(str(tmp_path)).eval()
    assert not type(oracle).__module__.startswith("gptqmodel")

    expected = _logits(oracle, inputs)
    actual = _logits(model, inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
