
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

from model_test import ModelTest


# | Metric                                             |   MARLIN |
# |----------------------------------------------------|----------|
# | arc_challenge :: acc,none                          |   0.3166 |
# | arc_challenge :: acc_norm,none                     |   0.3430 |
# | gsm8k_platinum_cot :: acc,num                      |   0.3906 |
# | mmlu_stem :: acc,none                              |   0.3942 |
class TestLlama3_2(ModelTest):
    # Keep one stable saved checkpoint so eval-only repro runs can reuse the exact post-quant model.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_SAVE_PATH",
        "/tmp/llama3_2_gptq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.3987,
                "floor_pct": 0.04,
            },
        },
        # "mmlu_stem": {
        #     "chat_template": False,
        #     "acc": {
        #         "value": 0.3860, # 0.3099 4096, 0.3270 2048
        #         "floor_pct": 0.04,
        #     },
        # },
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
        # "mmlu_stem": {
        #     "chat_template": False,
        #     "acc": {
        #         "value": 0.3942,
        #         "floor_pct": 0.04,
        #         "ceil_pct": 1.0,
        #     },
        #     "max_rows": 256,
        # },
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

    # llama 3.2 Instruct requires chat = true to have normal ARC scores
    # mmlu requires chat = false
    # APPLY_CHAT_TEMPLATE = True
    # QUANT_BATCH_SIZE = 4

    # EORA = Lora(
    #     # for quant, path is save path. for load, it is loading path
    #     path="./eora_test",
    #     rank=128,
    # )
    # b1 = 0.315, b4 = 0.3106, b8 = 0.3148, b32 = 0.3148, b16 = 0.3234

    def test_llama3_2(self):
        self.quantize_and_evaluate()
