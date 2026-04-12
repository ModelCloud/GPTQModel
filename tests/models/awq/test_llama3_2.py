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

from gptqmodel.quantization import FORMAT, METHOD


# | Metric                         | AWQ GEMM |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3140 |
# | arc_challenge :: acc_norm,none |   0.3541 |
# | mmlu_stem :: acc,none          |   0.3841 |
# | gsm8k_plat :: exact,flexible   |   0.3499 |
class TestLlama3_2_awq(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048 # new
    # STOP_AFTER_LAYER = 0
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.34987593052109184,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.31399317406143346,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.35409556313993173,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3840786552489692,
                "floor_pct": 0.04,
            },
        },
    }
    # Fast-mode regression scores captured on CUDA_VISIBLE_DEVICES=6 (RTX 4090).
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.4532671629445823,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.31313993174061433,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.35665529010238906,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3910561370123692,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ
    SYM = True
    TORCH_DTYPE = torch.float16

    def test_llama3_2_awq(self):
        self.quantize_and_evaluate()
