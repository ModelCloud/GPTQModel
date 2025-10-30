# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestKimiLinear(ModelTest):
    # FORMAT = FORMAT.GEMM
    # METHOD = METHOD.AWQ

    NATIVE_MODEL_ID = "/mnt/shared/Kimi-Linear-48B-A3B-Instruct"
    TRUST_REMOTE_CODE = True
    DELETE_QUANTIZED_MODEL = False
    DATASET_SIZE = 1024
    GROUP_SIZE = 32
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.5026, "floor_pct": 0.9},
            "acc_norm": {"value": 0.5171, "floor_pct": 0.9},
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "acc": {"value": 0.6362, "floor_pct": 0.9},
        },
    }
    def test_kimi_linear(self):
        self.quant_lm_eval()