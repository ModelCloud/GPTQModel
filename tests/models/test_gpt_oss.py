# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestGPTOSS(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-oss-20b-BF16/"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": False,
            "acc": {"value": 0.4411, "floor_pct": 0.2},
            "acc_norm": {"value": 0.4718, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False

    def test_gpt_oss(self):
        self.quant_lm_eval()
