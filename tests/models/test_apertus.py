# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.utils.eval import EVAL


class TestApertus(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Apertus-8B-Instruct-2509/"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5145, "floor_pct": 0.2},
            "acc_norm": {"value": 0.5256, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    EVAL_BATCH_SIZE = 6
    LOAD_BACKEND = BACKEND.TORCH

    def test_apertus(self):
        self.quant_lm_eval()
