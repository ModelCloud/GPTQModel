# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Original (non-HF) LLaVA checkpoints must be rejected with a clear message (#4229).

``model_type: llava`` covers both the Hugging Face ``LlavaForConditionalGeneration``
composite checkpoint and the original liuhaotian checkpoint. The latter used to be
routed to ``LlavaQModel`` and failed deep inside the shell builder with
"Padding_idx must be within num_embeddings".
"""

import json

import pytest

from gptqmodel.models.auto import _is_original_llava_config, check_and_get_model_definition
from gptqmodel.models.definitions.llava import LlavaQModel


ORIGINAL_LLAVA_CONFIG = {
    "model_type": "llava",
    "architectures": ["LlavaLlamaForCausalLM"],
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
}

HF_LLAVA_CONFIG = {
    "model_type": "llava",
    "architectures": ["LlavaForConditionalGeneration"],
    "text_config": {
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 100,
    },
    "vision_config": {
        "model_type": "clip_vision_model",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "image_size": 336,
        "patch_size": 14,
    },
}


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_original_llava_config_detected():
    config = _Config(**ORIGINAL_LLAVA_CONFIG)

    assert _is_original_llava_config(config) is True


def test_original_llava_config_detected_from_vision_tower_only():
    config = _Config(model_type="llava", mm_vision_tower="openai/clip-vit-large-patch14-336")

    assert _is_original_llava_config(config) is True


def test_hf_llava_config_not_detected():
    config = _Config(**HF_LLAVA_CONFIG)

    assert _is_original_llava_config(config) is False


def test_llava_config_without_architectures_not_detected():
    config = _Config(model_type="llava")

    assert _is_original_llava_config(config) is False


def _write_config(tmp_path, payload):
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path)


def test_dispatch_rejects_original_llava_checkpoint(tmp_path):
    model_dir = _write_config(tmp_path, ORIGINAL_LLAVA_CONFIG)

    with pytest.raises(ValueError, match="llava-hf/llava-1.5-7b-hf"):
        check_and_get_model_definition(model_dir)


def test_dispatch_keeps_hf_llava_checkpoint(tmp_path):
    model_dir = _write_config(tmp_path, HF_LLAVA_CONFIG)

    assert check_and_get_model_definition(model_dir) is LlavaQModel
