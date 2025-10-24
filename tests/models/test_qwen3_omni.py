# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestQwen3Omni(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Omni-30B-A3B-Instruct/"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.2739, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3055, "floor_pct": 0.2},
        },
    }
    # # TRUST_REMOTE_CODE = False
    # APPLY_CHAT_TEMPLATE = True
    # # EVAL_BATCH_SIZE = 6
    # V2 = False
    # DEBUG = True
    # ACT_GROUP_AWARE = True
    # DESC_ACT = False
    # DATASET_SIZE = 1024
    # DATASET_SORT = "desc"
    QUANT_BATCH_SIZE = 1

    def test_omni(self):
        self.quant_lm_eval()
