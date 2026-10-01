# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


class TestGraniteMoE(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/PowerMoE-3b" # ibm-research/PowerMoE-3b
    TRUST_REMOTE_CODE = False
    DATASET_SIZE_FAST = 128
    EVAL_BATCH_SIZE = 8
    USE_FLASH_ATTN = False

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.3874, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4164, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    # Exercise every expert during calibration so sparse routing does not
    # leave low-traffic expert projections without Hessian observations.
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_granitemoe(self):
        self.quantize_and_evaluate()
