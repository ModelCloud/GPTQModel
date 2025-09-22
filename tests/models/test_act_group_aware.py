# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestHybridActOrder(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3140 # A100
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3439 # A100
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.10
    APPLY_CHAT_TEMPLATE = True
    V2 = False
    ACT_GROUP_AWARE = True

    def test_llama3_2(self):
        self.quant_lm_eval()
