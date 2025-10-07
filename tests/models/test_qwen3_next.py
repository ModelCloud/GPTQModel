# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestQwen3Next(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-Next-80B-A3B-Instruct"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2739
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3055
    TRUST_REMOTE_CODE = True
    APPLY_CHAT_TEMPLATE = True
    EVAL_BATCH_SIZE = 6
    #DATASET_SIZE = 1024

    def test_mimo(self):
        self.quant_lm_eval()
