# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

from model_test import ModelTest


class TestLlama3_2AdaptiveClipping(ModelTest):
    # Use a separate save path so this test does not load the default checkpoint.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_ADAPTIVE_CLIPPING_SAVE_PATH",
        "/tmp/llama3_2_gptq_adaptive_clipping_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048

    # Same baseline expectations as the default Llama-3.2-1B test.  Adaptive
    # clipping should meet or exceed these scores.
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

    ADAPTIVE_CLIPPING = {
        "enabled": True,
        "metric": "gptq_error",
        "per_group": True,
    }

    def test_llama3_2_adaptive_clipping(self):
        self.quantize_and_evaluate()
