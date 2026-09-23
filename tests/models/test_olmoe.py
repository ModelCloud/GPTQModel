# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


class TestOlmoE(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/OLMoE-1B-7B-0924" # allenai/OLMoE-1B-7B-0924
    TRUST_REMOTE_CODE = False
    DATASET_SIZE_FAST = 128
    EVAL_BATCH_SIZE = 8
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.4710, "floor_pct": 0.25},
            "acc_norm": {"value": 0.4932, "floor_pct": 0.25},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    USE_FLASH_ATTN = False

    def test_olmoe(self):
        self.quantize_and_evaluate()
