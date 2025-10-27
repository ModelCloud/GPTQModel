# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestXVerse(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/XVERSE-7B-Chat" # "xverse/XVERSE-7B-Chat"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.4198, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4044, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False
    USE_FLASH_ATTN = False

    def test_xverse(self):
        self.quant_lm_eval()
