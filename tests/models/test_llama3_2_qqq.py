# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT, METHOD


# QQQ equivalent of tests/models/test_llama3_2.py.
#
# Fast-mode post-quant baselines (eager attention, group_size=128, 4bit QQQ):
#   gsm8k_platinum_cot :: acc,num          0.4574
#   arc_challenge        :: acc            0.3225
#   arc_challenge        :: acc_norm       0.3541
class TestLlama3_2_QQQ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_QQQ_SAVE_PATH",
        "/tmp/llama3_2_qqq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    USE_FLASH_ATTN = False

    METHOD = METHOD.QQQ
    FORMAT = FORMAT.QQQ
    LOAD_BACKEND = BACKEND.QQQ
    QUANT_BACKEND = BACKEND.AUTO
    BITS = 4
    GROUP_SIZE = 128
    DESC_ACT = False
    SYM = True
    ACT_GROUP_AWARE = False

    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "eager",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": 0.4574028122415219,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3225255972696246,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.35409556313993173,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.4574028122415219,
                "floor_pct": 0.10,
                "ceil_pct": 0.10,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3225255972696246,
                "floor_pct": 0.10,
                "ceil_pct": 0.10,
            },
            "acc_norm": {
                "value": 0.35409556313993173,
                "floor_pct": 0.10,
                "ceil_pct": 0.10,
            },
        },
    }

    def test_llama3_2_qqq(self):
        self.quantize_and_evaluate()
