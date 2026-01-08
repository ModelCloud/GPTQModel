# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.eval import EVAL


class TestGlm4Moe(ModelTest):
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/monster/data/model/GLM-4.6/"
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 512
    GROUP_SIZE = 32
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5026, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5171, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.6362, "floor_pct": 0.04},
        },
    }
    def test_glm4moe(self):
        self.quant_lm_eval()


class TestGlm4_5_Air(ModelTest):
    FORMAT = FORMAT.GEMM
    METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/monster/data/model/GLM-4.5-Air/"
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 512
    GROUP_SIZE = 32
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5247, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5614, "floor_pct": 0.04},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.6403, "floor_pct": 0.04},
        },
    }

    def test_glm4moe(self):
        self.quant_lm_eval()
