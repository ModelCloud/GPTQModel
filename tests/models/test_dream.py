# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestDream(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Dream-v0-Instruct-7B"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.3567, "floor_pct": 0.36},
            "acc_norm": {"value": 0.3805, "floor_pct": 0.36},
        },
    }
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1
    BITS = 8

    def test_dream(self):
        self.quant_lm_eval()
