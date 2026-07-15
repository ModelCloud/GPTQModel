# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestNemotronHPuzzle(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16" # nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False

    # MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))

    # The first two layers cover both Mamba and Puzzle's two-projection MoE.
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4880, "floor_pct": 0.36},
            "acc_norm": {"value": 0.4462, "floor_pct": 0.36},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    MODEL_COMPAT_FAST_LAYER_COUNT = 10
    EVAL_SINGLE_GPU = False

    def test_nemotron_ultra(self):
        self.quantize_and_evaluate()
