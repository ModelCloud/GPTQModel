# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization import FORMAT, METHOD


class TestQwen3_8B_Base_AWQ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-8B-Base"  # "Qwen/Qwen3-8B-Base"
    EVAL_TASKS = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.2994,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
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
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3692,
                "floor_pct": 0.04,
            },
        },
    }
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ
    QUANT_BATCH_SIZE = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_qwen3_8b_base_awq(self):
        self.quantize_and_evaluate()
