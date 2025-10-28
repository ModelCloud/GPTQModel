# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.eval import EVAL


class TestLongLlama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/long_llama_3b_instruct" # "syzymon/long_llama_3b_instruct"
    TRUST_REMOTE_CODE = True
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.3515, "floor_pct": 0.5},
            "acc_norm": {"value": 0.3652, "floor_pct": 0.5},
        },
    }
    USE_VLLM = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.TORCH

    def test_longllama(self):
        self.quant_lm_eval()
