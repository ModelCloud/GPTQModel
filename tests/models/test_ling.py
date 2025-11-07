# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestLing(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ling-mini-2.0/"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.5009, "floor_pct": 0.2},
            "acc_norm": {"value": 0.5137, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = True
    # EVAL_BATCH_SIZE = 6
    GPTQA = False
    DEBUG = True
    ACT_GROUP_AWARE = True
    DESC_ACT = False
    DATASET_SIZE = 2048
    DATASET_SORT = "desc"
    QUANT_BATCH_SIZE = 8
    CALIB_NOISE_MODE = "unseen"
    CALIB_NOISE_PERCENT = 0.025

    def test_mimo(self):
        self.quant_lm_eval()
