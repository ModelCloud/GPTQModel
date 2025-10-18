# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest
from gptqmodel.utils.eval import EVAL


class TestGlm(ModelTest):
    # real: THUDM/glm-4-9b-chat-hf
    NATIVE_MODEL_ID = "/monster/data/model/glm-4-9b-chat-hf"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5154, "floor_pct": 0.2},
            "acc_norm": {"value": 0.5316, "floor_pct": 0.2},
        },
    }
    USE_VLLM = False

    def test_glm(self):
        self.quant_lm_eval()
