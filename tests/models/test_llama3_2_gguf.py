# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


class TestLlama3_2_GGUF(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"

    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.3871,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3955,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3106,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3532,
                "floor_pct": 0.04,
                "ceil_pct": 0.04,
            },
        },
    }
    METHOD = METHOD.GGUF
    FORMAT = FORMAT.GGUF
    BITS = "q4_k_m"
    LOAD_BACKEND = BACKEND.GGUF_TORCH
    KERNEL_INFERENCE = {GGUFTorchQuantLinear}

    def test_llama3_2_gguf_full_model(self):
        self.quant_lm_eval()
