# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, Fallback, MoEConfig, VramStrategy


LAST_FOUR_ONLY_NEGATIVE_MATCH = r"^model\.language_model\.layers\.(?:[0-9]|[1-2][0-9]|3[0-5])(?:\.|$)"


def _resolve_qwen3_6_moe_model_path() -> str:
    """Prefer the user-requested local-dir path, but allow an explicit override for repro runs."""

    override = os.environ.get("GPTQMODEL_QWEN3_6_MOE_MODEL_PATH")
    if override:
        return override

    requested = "/moonster/data/model/Qwen3.6-35B-A3B"
    legacy_fallback = "/monster/data/model/Qwen3.6-35B-A3B"
    if os.path.isdir(requested):
        return requested
    if os.path.isdir(legacy_fallback):
        return legacy_fallback
    return requested


class TestQwen3_6Moe(ModelTest):
    """Fast 3-A100 compat regression for the Qwen 3.6 MoE checkpoint released on the Qwen hub."""

    FALLBACK = Fallback()
    NATIVE_MODEL_ID = _resolve_qwen3_6_moe_model_path()
    DATASET_SIZE = 16
    DATASET_CONCAT_SIZE = 1024
    QUANT_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = "auto"
    # Keep post-quant validation spread across the visible A100 pool instead of forcing a single-GPU reload.
    EVAL_SINGLE_GPU = False
    # The native checkpoint has 40 decoder layers; skip 0-35 so only the last four are quantized.
    DYNAMIC = {
        f"-:{LAST_FOUR_ONLY_NEGATIVE_MATCH}": {},
    }
    DENSE_VRAM_STRATEGY = VramStrategy.EXCLUSIVE
    # Keep the dense shell on the first visible device and spread expert work across the rest.
    DENSE_VRAM_STRATEGY_DEVICES = ["cuda:0"]
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    # The mixed RTX 4090 slot on this host OOMs during late replay for this 35B MoE path,
    # so keep the quantization pool on the A100-class devices.
    MOE_VRAM_STRATEGY_DEVICES = ["cuda:1", "cuda:2"]
    # Fast compat mode already trims the decoder stack; keep routing aligned with the checkpoint default.
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=8))
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": False,
            # Evaluate the already-loaded post-quant wrapper so the three-A100 placement survives into Evalution.
            "evalution_batch_size": 4,
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device_map": "auto",
            },
            "evalution_suite_kwargs": {
                "batch_size": 4,
                "max_new_tokens": 256,
                "stream": True,
                "max_rows": 8,
            },
            "acc,num": {
                "value": {
                    # Recorded on 2026-04-16 with three visible A100s (dense=cuda:0, MoE=cuda:1,cuda:2).
                    "A100": 1.0,
                },
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    def test_qwen3_6_moe(self):
        self.quantize_and_evaluate()
