# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

# a100:0
# desc_act = False, act_group_aware = False 0.2500/0.2841
# desc_act = False, act_group_aware = True 0.3063/0.3456
# desc_act = True, 0.3089/0.3328
class TestLlama3_2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36
    APPLY_CHAT_TEMPLATE = True
    V2 = False
    DEBUG = True
    ACT_GROUP_AWARE = True
    DESC_ACT = False

    def test_llama3_2(self):
        self.quant_lm_eval()
