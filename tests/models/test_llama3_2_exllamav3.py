# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import sys

import torch


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


# gpu0/a100
# | Metric                                             | EXLLAMA_V3 |
# |----------------------------------------------------|------------|
# | arc_challenge :: acc,none                          |     0.3174 |
# | arc_challenge :: acc_norm,none                     |     0.3456 |
# | gsm8k_platinum_cot :: exact_match,flexible-extract |     0.4715 |
# | gsm8k_platinum_cot :: exact_match,strict-match     |     0.4218 |
# | mmlu_stem :: acc,none                              |     0.3977 |
class TestLlama3_2_ExllamaV3(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.4715,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3977,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3174,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3456,
                "floor_pct": 0.04,
            },
        },
    }

    FORMAT = FORMAT.EXL3
    METHOD = METHOD.EXL3
    BITS = 4.0
    GROUP_SIZE = -1
    ACT_GROUP_AWARE = False
    TORCH_DTYPE = torch.float16
    QUANT_BACKEND = BACKEND.EXLLAMA_V3
    LOAD_BACKEND = BACKEND.EXLLAMA_V3
    PIN_CUDA_DEVICE = 0

    def test_llama3_2_exllamav3(self):
        self.quant_lm_eval()
