# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT, METHOD
from model_test import ModelTest


# E2E Swordfish (Blackwell sm100/sm110) integration test using the same
# Llama-3.2-1B model harness as test_llama3_2.py.
@unittest.skipIf(
    os.environ.get("GPTQMODEL_SKIP_SWORDFISH_E2E", "0").lower() in {"1", "true", "yes"},
    "Swordfish e2e test disabled by GPTQMODEL_SKIP_SWORDFISH_E2E",
)
class TestLlama3_2_Swordfish(ModelTest):
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_SWORDFISH_SAVE_PATH",
        "/tmp/llama3_2_gptq_swordfish_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048

    LOAD_BACKEND = BACKEND.GPTQ_SWORDFISH
    QUANT_BACKEND = BACKEND.AUTO
    BITS = 4
    GROUP_SIZE = 128
    DESC_ACT = False
    SYM = True
    FORMAT = FORMAT.GPTQ
    METHOD = METHOD.GPTQ

    # Accuracy expectations mirror the Marlin baseline in test_llama3_2.py.
    # Swordfish should be bit-close on the same calibration seed.
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3140,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3507,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": 0.4690,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3987,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3234,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3643,
                "floor_pct": 0.04,
            },
        },
    }

    def test_llama3_2_swordfish(self):
        self.quantize_and_evaluate()
