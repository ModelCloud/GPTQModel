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


# Full-model ARC quality gates used by the AWQ evaluation.
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.5094 |
# | arc_challenge :: acc_norm,none |   0.5486 |
#
# H100 snapshot measurement (32 calibration rows, q-projections quantized,
# eager attention, 1,172 ARC-Challenge test samples):
# |--------------------------------|----------|
# | accuracy,loglikelihood         | 0.5273037542662116 |
# | accuracy,loglikelihood_norm    | 0.552901023890785  |
# These are recorded results, not replacements for the full-model gates above.
H100_ARC_SNAPSHOT_RESULTS = {
    "accuracy,loglikelihood": 0.5273037542662116,
    "accuracy,loglikelihood_norm": 0.552901023890785,
}


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

    def test_h100_arc_snapshot_scores_clear_quality_gates(self):
        """Keep the measured H100 snapshot score regression reference explicit."""

        self.assertGreaterEqual(
            H100_ARC_SNAPSHOT_RESULTS["accuracy,loglikelihood"],
            self.EVAL_TASKS_SLOW["arc_challenge"]["acc"]["value"],
        )
        self.assertGreaterEqual(
            H100_ARC_SNAPSHOT_RESULTS["accuracy,loglikelihood_norm"],
            self.EVAL_TASKS_SLOW["arc_challenge"]["acc_norm"]["value"],
        )

    def test_moe_awq(self):
        self.quantize_and_evaluate()
