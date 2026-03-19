# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


class TestLlama3_2_ParoQuant(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.3391232423490488,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3166,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3464,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3783698065334602,
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    FORMAT = FORMAT.PAROQUANT
    METHOD = METHOD.PAROQUANT
    SYM = True
    TORCH_DTYPE = torch.float16
    LOAD_BACKEND = BACKEND.PAROQUANT
    QUANT_BACKEND = BACKEND.PAROQUANT
    KERNEL_QUANT = {ParoQuantQuantLinear}
    KERNEL_INFERENCE = {ParoQuantQuantLinear}

    def test_llama3_2_paroquant(self):
        self.quant_lm_eval()
