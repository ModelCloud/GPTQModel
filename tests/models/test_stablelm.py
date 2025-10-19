# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestStablelm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/stablelm-base-alpha-3b"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.2363, "floor_pct": 0.2},
            "acc_norm": {"value": 0.2577, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6

    def test_stablelm(self):
        self.quant_lm_eval()
