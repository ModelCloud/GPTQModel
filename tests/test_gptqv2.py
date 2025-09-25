# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from models.model_test import ModelTest


class TestQwen2_5_GPTQv2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2739
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3055
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    #EVAL_BATCH_SIZE = 6
    V2 = True

    def test_qwen2_5(self):
        self.quant_lm_eval()
