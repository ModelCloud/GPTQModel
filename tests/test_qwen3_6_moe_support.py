# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from transformers import AutoConfig

from gptqmodel.models.auto import check_and_get_model_definition
from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel


def test_qwen3_6_moe_reuses_the_qwen3_5_moe_transformers_definition(tmp_path):
    """Guard the real Qwen 3.6 MoE config shape shipped on the Hub."""

    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5_moe",
        "text_config": {
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "hidden_size": 2048,
            "layer_types": layer_types,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 512,
            "num_attention_heads": 16,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 40,
            "num_key_value_heads": 2,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 248320,
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.57.1",
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_size": 1152,
            "in_channels": 3,
            "intermediate_size": 4304,
            "model_type": "qwen3_5_moe",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }
    model_dir = tmp_path / "qwen3_6_moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    resolved_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    model_definition = check_and_get_model_definition(model_dir, trust_remote_code=False)

    assert type(resolved_config).__name__ == "Qwen3_5MoeConfig"
    assert model_definition is Qwen3_5_MoeQModel
