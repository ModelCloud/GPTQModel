# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestHymba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Hymba-1.5B-Instruct/"  # "baichuan-inc/Baichuan2-7B-Chat"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.2073, "floor_pct": 0.75},
            "acc_norm": {"value": 0.2713, "floor_pct": 0.75},
        },
    }
    MODEL_MAX_LEN = 8192
    TRUST_REMOTE_CODE = True
    # Hymba currently only supports a batch size of 1.
    # See https://huggingface.co/nvidia/Hymba-1.5B-Instruct
    EVAL_BATCH_SIZE = 1

    # Hymba currently tests that DESC_ACT=False to get better results.
    # If DESC_ACT=False, the output will be terrible.
    DESC_ACT = False


    def test_hymba(self):
        self.quant_lm_eval()
