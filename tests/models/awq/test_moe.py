# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_test import ModelTest

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.5094 |
# | arc_challenge :: acc_norm,none |   0.5486 |
class TestQwen3MoeAwq(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-30B-A3B"
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5094, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5486, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride())

    def test_moe_awq(self):
        self.quantize_and_evaluate()
