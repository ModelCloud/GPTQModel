# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.bitsandbytes import BITSANDBYTES_AVAILABLE, BitsAndBytesLinear
from gptqmodel.quantization import FORMAT, METHOD


class TestLlama3_2_BitsAndBytes(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.31,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.34,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.36,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.39,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.31,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.34,
                "floor_pct": 0.04,
            },
        },
    }

    METHOD = METHOD.BITSANDBYTES
    FORMAT = FORMAT.BITSANDBYTES
    BITS = 4
    GROUP_SIZE = -1
    LOAD_BACKEND = BACKEND.BITSANDBYTES
    QUANT_BACKEND = BACKEND.BITSANDBYTES
    KERNEL_QUANT = {BitsAndBytesLinear}
    KERNEL_INFERENCE = {BitsAndBytesLinear}
    BNB_BLOCK_SIZE = 64
    BNB_COMPRESS_STATISTICS = False


    def test_llama3_2_bitsandbytes(self):
        if not BITSANDBYTES_AVAILABLE:
            self.skipTest("bitsandbytes backend unavailable")
        self.quantize_and_evaluate()

        module = self.model.model.model.layers[0].self_attn.q_proj
        assert isinstance(module, BitsAndBytesLinear)
        for name in ("weight", "weight_scb"):
            assert hasattr(module, name), f"missing `{name}`"
        assert tuple(module.weight.shape) == (24, 48)
        assert tuple(module.weight_scb.shape) == (24,)
        assert module.weight.dtype == torch.int8
        assert module.weight_scb.dtype == torch.float32
