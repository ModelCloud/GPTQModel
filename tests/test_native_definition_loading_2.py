# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Group-2 definitions load vendored Transformers 5.23 classes through their public loader.

For every model type, a tiny config built with the *vendored* config class seeds a
checkpoint saved by the native Transformers class (the oracle; tests may import
``transformers.models``). The checkpoint is loaded through
``MODEL_MAP[model_type].loader``, must contain only vendored modules, and must
match the oracle's own ``from_pretrained`` of the same checkpoint exactly.
"""

import importlib
import json
from pathlib import Path

import pytest
import torch
from eager_ssm_kernels import eager_torch_ssm_kernels
from safetensors.torch import load_file

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader


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


def _vendored(family, module, name):
    return getattr(importlib.import_module(f"{NATIVE}{family}.{module}"), name)


def _oracle(family, module, name):
    return getattr(importlib.import_module(f"transformers.models.{family}.{module}"), name)


def _fixture(family):
    return json.loads((FIXTURES / f"{family}.json").read_text())


def _fixture_inputs(family, *, text_only=False):
    spec = _fixture(family)
    tensors = {}
    path = FIXTURES / f"{family}.safetensors"
    if path.exists():
        tensors.update(load_file(str(path)))
    generator = torch.Generator().manual_seed(0)
    for key in sorted(spec["float_inputs"]):
        meta = spec["float_inputs"][key]
        tensors[key] = torch.randn(meta["shape"], generator=generator).to(getattr(torch, meta["dtype"]))
    tensors.update(spec["extra_inputs"])
    if text_only:
        tensors = {key: tensors[key] for key in ("input_ids", "attention_mask") if key in tensors}
    return tensors


def _fixture_config(family, *, sub=None, config_class=None):
    spec = _fixture(family)
    payload = dict(spec["config"])
    if sub is not None:
        payload = dict(payload[sub])
    name = config_class or spec["config_class"]
    return _vendored(family, spec["config_module"], name).from_dict(payload)


def _text_ids(vocab_size, batch=2, length=9):
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(3, vocab_size, (batch, length), generator=generator)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


# ---------------------------------------------------------------------------
# Hand-written cases: no Transformers model tester fixture (llama4, opt) or the
# definition binds a class the fixture does not build.
# ---------------------------------------------------------------------------


def _opt_case():
    config = _vendored("opt", "configuration_opt", "OPTConfig")(
        vocab_size=99, hidden_size=16, num_hidden_layers=2, ffn_dim=32, num_attention_heads=4,
        max_position_embeddings=64, word_embed_proj_dim=16, pad_token_id=1, bos_token_id=2, eos_token_id=2,
    )
    return config, _text_ids(99)


def _llama4_text_config():
    return _vendored("llama4", "configuration_llama4", "Llama4TextConfig")(
        vocab_size=99, hidden_size=32, intermediate_size=64, intermediate_size_mlp=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=4, num_experts_per_tok=1,
        interleave_moe_layer_step=2, no_rope_layers=[1, 1, 1, 0], attention_chunk_size=4,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )


def _llama4_text_case():
    return _llama4_text_config(), _text_ids(99)


def _llama4_case():
    image_token = 98
    vision = _vendored("llama4", "configuration_llama4", "Llama4VisionConfig")(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, image_size=16,
        patch_size=4, vision_output_dim=32, projector_input_dim=32, projector_output_dim=32,
        pixel_shuffle_ratio=0.5,
    )
    config = _vendored("llama4", "configuration_llama4", "Llama4Config")(
        text_config=_llama4_text_config().to_dict(), vision_config=vision.to_dict(), image_token_index=image_token,
    )
    # 16x16 image, 4x4 patches -> 16 patches -> pixel shuffle 0.5 -> 4 image tokens per image.
    inputs = _text_ids(90, batch=1, length=12)
    inputs["input_ids"][0, 2:6] = image_token
    inputs["pixel_values"] = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    return config, inputs


def _gemma4_text_case():
    return _fixture_config("gemma4", sub="text_config", config_class="Gemma4TextConfig"), _fixture_inputs("gemma4", text_only=True)


def _qwen4_exp_case():
    module = "configuration_qwen4_exp"
    text = _fixture_config("qwen4_exp")
    vision = _vendored("qwen4_exp", module, "Qwen4ExpVisionConfig")(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=4, spatial_merge_size=2,
        temporal_patch_size=2, out_hidden_size=text.hidden_size, num_position_embeddings=16,
    )
    config = _vendored("qwen4_exp", module, "Qwen4ExpConfig")(text_config=text.to_dict(), vision_config=vision.to_dict())
    return config, _fixture_inputs("qwen4_exp", text_only=True)


def _qwen2_5_omni_case():
    thinker = _fixture_config("qwen2_5_omni")
    config = _vendored("qwen2_5_omni", "configuration_qwen2_5_omni", "Qwen2_5OmniConfig")(
        thinker_config=thinker.to_dict(), enable_audio_output=False,
    )
    return config, _fixture_inputs("qwen2_5_omni")


def _fixture_case(family, *, text_only=False):
    return lambda: (_fixture_config(family), _fixture_inputs(family, text_only=text_only))


# model_type -> (family, native modeling module, class the definition loader must build, case factory)
CASES = {
    "cohere2_moe": ("cohere2_moe", "Cohere2MoeForCausalLM", _fixture_case("cohere2_moe")),
    "deepseek_ocr2": ("deepseek_ocr2", "DeepseekOcr2ForConditionalGeneration", _fixture_case("deepseek_ocr2")),
    "diffusion_gemma": ("diffusion_gemma", "DiffusionGemmaForBlockDiffusion", _fixture_case("diffusion_gemma")),
    "ernie4_5_moe": ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM", _fixture_case("ernie4_5_moe")),
    "flint": ("flint", "FlintForCausalLM", _fixture_case("flint")),
    "gemma4": ("gemma4", "Gemma4ForConditionalGeneration", _fixture_case("gemma4")),
    "gemma4_text": ("gemma4", "Gemma4ForCausalLM", _gemma4_text_case),
    "glm4v": ("glm4v", "Glm4vForConditionalGeneration", _fixture_case("glm4v")),
    # glm4v_moe inherits Glm4vGPTQ's loader.
    "glm4v_moe": ("glm4v_moe", "Glm4vMoeForConditionalGeneration", _fixture_case("glm4v_moe")),
    "glmasr": ("glmasr", "GlmAsrForConditionalGeneration", _fixture_case("glmasr")),
    "gpt_neox": ("gpt_neox", "GPTNeoXForCausalLM", _fixture_case("gpt_neox")),
    "granitemoe": ("granitemoe", "GraniteMoeForCausalLM", _fixture_case("granitemoe")),
    "hunyuan_v1_dense": ("hunyuan_v1_dense", "HunYuanDenseV1ForCausalLM", _fixture_case("hunyuan_v1_dense")),
    "inkling_mm_model": ("inkling", "InklingForConditionalGeneration", _fixture_case("inkling")),
    "lfm2": ("lfm2", "Lfm2ForCausalLM", _fixture_case("lfm2")),
    "llama4": ("llama4", "Llama4ForConditionalGeneration", _llama4_case),
    "llama4_text": ("llama4", "Llama4ForCausalLM", _llama4_text_case),
    "llava": ("llava", "LlavaForConditionalGeneration", _fixture_case("llava")),
    "minicpmv4_6": ("minicpmv4_6", "MiniCPMV4_6ForConditionalGeneration", _fixture_case("minicpmv4_6")),
    "mistral3": ("mistral3", "Mistral3ForConditionalGeneration", _fixture_case("mistral3")),
    "olmo2": ("olmo2", "Olmo2ForCausalLM", _fixture_case("olmo2")),
    "opt": ("opt", "OPTForCausalLM", _opt_case),
    "qwen2_5_omni": ("qwen2_5_omni", "Qwen2_5OmniForConditionalGeneration", _qwen2_5_omni_case),
    "qwen3_5_moe": ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration", _fixture_case("qwen3_5_moe")),
    "qwen3_next": ("qwen3_next", "Qwen3NextForCausalLM", _fixture_case("qwen3_next")),
    "qwen4_exp": ("qwen4_exp", "Qwen4ExpForConditionalGeneration", _qwen4_exp_case),
    "qwen4_exp_text": ("qwen4_exp", "Qwen4ExpForCausalLM", _fixture_case("qwen4_exp")),
    "starcoder2": ("starcoder2", "Starcoder2ForCausalLM", _fixture_case("starcoder2")),
}

PROCESSOR_MODEL_TYPES = {
    "deepseek_ocr2", "diffusion_gemma", "glmasr", "inkling_mm_model", "llava", "minicpmv4_6", "qwen2_5_omni",
    "qwen3_5_moe", "qwen4_exp",
}


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


def _forward(model, model_type, inputs):
    # Qwen2.5-Omni's waveform wrapper has no forward; the definition runs the thinker.
    target = model.thinker if model_type == "qwen2_5_omni" else model
    with torch.no_grad():
        return target(**inputs)


@pytest.mark.parametrize("model_type", sorted(CASES))
def test_definition_loader_builds_vendored_classes_matching_transformers(model_type, tmp_path):
    if model_type == "inkling_mm_model":
        pytest.importorskip("scipy", reason="Inkling modeling (vendored and native) requires scipy")
    family, class_name, case = CASES[model_type]
    loader = MODEL_MAP[model_type].loader
    assert isinstance(loader, LocalModelLoader)
    assert (loader.processor_class is not None) == (model_type in PROCESSOR_MODEL_TYPES)

    config, inputs = case()
    assert type(config).__module__.startswith(NATIVE)
    assert config.model_type == model_type

    # The checkpoint is written by the native Transformers class, so loading it
    # through the definition exercises Transformers' checkpoint conversion mappings.
    modeling = f"modeling_{family}"
    oracle_class = _oracle(family, modeling, class_name)
    native_config = oracle_class.config_class.from_dict(config.to_dict())
    assert type(native_config).__module__.startswith("transformers.models.")
    torch.manual_seed(0)
    native = oracle_class._from_config(native_config, dtype=torch.float32)
    checkpoint = tmp_path / "checkpoint"
    native.save_pretrained(checkpoint)
    if model_type == "qwen2_5_omni":
        torch.save({}, checkpoint / "spk_dict.pt")
    del native

    loaded_config = loader.load_config(str(checkpoint))
    assert type(loaded_config) is type(config)
    loaded = loader.from_pretrained(str(checkpoint), config=loaded_config, dtype=torch.float32).eval()

    assert type(loaded) is _vendored(family, modeling, class_name)
    foreign = sorted({type(m).__module__ for m in loaded.modules() if not type(m).__module__.startswith(NATIVE)})
    leaked = [name for name in foreign if name.startswith("transformers.models")]
    assert not leaked, f"{model_type}: Transformers model-family modules leaked: {leaked}"
    # Only torch layers and Transformers framework infrastructure (e.g. activations) may be foreign.
    assert all(name.startswith(("torch.", "transformers.")) for name in foreign), f"{model_type}: {foreign}"
    configs = [loaded.config, *(getattr(m, "config", None) for m in loaded.modules())]
    assert all(type(c).__module__.startswith(NATIVE) for c in configs if c is not None)

    oracle = oracle_class.from_pretrained(str(checkpoint), dtype=torch.float32).eval()
    assert type(oracle).__module__.startswith("transformers.models.")
    loaded_state, oracle_state = loaded.state_dict(), oracle.state_dict()
    assert list(loaded_state) == list(oracle_state)
    for key, tensor in oracle_state.items():
        assert torch.equal(loaded_state[key], tensor), f"{model_type}: {key} differs"

    expected = _flatten(_forward(oracle, model_type, inputs))
    actual = _flatten(_forward(loaded, model_type, inputs))
    assert expected and len(actual) == len(expected)
    for index, (want, got) in enumerate(zip(expected, actual)):
        assert torch.equal(want, got), f"{model_type}: output {index} differs"
