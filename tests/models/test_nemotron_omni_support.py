
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from model_test import ModelTest


class Test(ModelTest):
    # Keep one stable saved checkpoint so eval-only repro runs can reuse the exact post-quant model.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_NEMOTRON_3_SAVE_PATH",
        "/tmp/Nemotron_3_gptq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"  # nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    TRUST_REMOTE_CODE = True
    # TODO, update scores later
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
                "value": 0.3234,  # 0.3294 4096, 0.3242 2048
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3643,  # 0.3558 4096, 0.3635 2048
                "floor_pct": 0.04,
            },
        },
    }
    EVAL_TASKS_FAST = {
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
                "value": 0.390625,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3166,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3430,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }

    def test(self):
        self.quantize_and_evaluate()
