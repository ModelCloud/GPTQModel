# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch


TESTS_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TESTS_MODELS_ROOT not in sys.path:
    sys.path.insert(0, TESTS_MODELS_ROOT)

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization import METHOD
from gptqmodel.quantization.config import WeightOnlyConfig


# | Metric                                             | TORCH_FP8 |
# |----------------------------------------------------|-----------|
# | arc_challenge :: acc,none                          |    0.3191 |
# | arc_challenge :: acc_norm,none                     |    0.3498 |
# | gsm8k_platinum_cot :: acc,num |    0.4756 |
# | gsm8k_platinum_cot :: exact_match,strict-match     |    0.4458 |
# | mmlu_stem :: acc,none                              |    0.4085 |
class TestLlama3_2_FP8(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.4756,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.4085,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3191,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3498,
                "floor_pct": 0.04,
            },
        },
    }

    FORMAT = "float8_e4m3fn"
    METHOD = METHOD.FP8
    BITS = 8
    GROUP_SIZE = -1
    ACT_GROUP_AWARE = False
    TORCH_DTYPE = torch.float16
    WEIGHT_ONLY = WeightOnlyConfig(method="fp8")
    QUANT_BACKEND = BACKEND.TORCH
    LOAD_BACKEND = BACKEND.TORCH
    KERNEL_QUANT = {TorchFP8Linear}
    KERNEL_INFERENCE = {TorchFP8Linear}

    def test_llama3_2_fp8(self):
        self.quantize_and_evaluate()
