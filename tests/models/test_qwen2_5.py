# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.2892 |
# | arc_challenge :: acc_norm,none |   0.3302 |
# | mmlu_stem :: acc,none          |   0.4351 |
class TestQwen2_5(ModelTest):
    GROUP_SIZE = 32
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
    EVAL_BATCH_SIZE = 64
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.2910, "floor_pct": 0.04},
            "acc_norm": {"value": 0.3268, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.3819, "floor_pct": 0.04},
        },
    }

    def test_qwen2_5(self):
        self.quant_lm_eval()
