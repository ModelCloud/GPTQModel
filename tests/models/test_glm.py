# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.5154 |
# | arc_challenge :: acc_norm,none |   0.535  |
# | mmlu_stem :: acc,none          |   0.6325 |
class TestGlm(ModelTest):
    GROUP_SIZE = 32
    # real: THUDM/glm-4-9b-chat-hf
    NATIVE_MODEL_ID = "/monster/data/model/glm-4-9b-chat-hf"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5154, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5350, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.6325, "floor_pct": 0.04},
        },
    }

    def test_glm(self):
        self.quant_lm_eval()
