# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.bitsandbytes import BITSANDBYTES_AVAILABLE, BitsAndBytesQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL
import torch


class TestLlama3_2_BitsAndBytes(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
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
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.36,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.39,
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
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
    FORMAT = "fp4"
    BITS = 4
    GROUP_SIZE = -1
    LOAD_BACKEND = BACKEND.BITSANDBYTES
    QUANT_BACKEND = BACKEND.BITSANDBYTES
    KERNEL_QUANT = {BitsAndBytesQuantLinear}
    KERNEL_INFERENCE = {BitsAndBytesQuantLinear}

    def test_llama3_2_bitsandbytes(self):
        if not BITSANDBYTES_AVAILABLE:
            self.skipTest("bitsandbytes backend unavailable")
        self.quant_lm_eval()

        module = self.model.model.model.layers[0].self_attn.q_proj
        assert module.QUANT_TYPE == "bitsandbytes"
        for name in ("weight", "weight_scb"):
            assert hasattr(module, name), f"missing `{name}`"
        assert tuple(module.weight.shape) == (24, 48)
        assert tuple(module.weight_scb.shape) == (24,)
        assert module.weight.dtype == torch.int8
        assert module.weight_scb.dtype == torch.float32