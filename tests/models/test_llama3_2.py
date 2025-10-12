# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# a100:7
# desc_act = False, act_group_aware = False 0.2918/0.3422
# desc_act = False, act_group_aware = True 0.3311/0.3549
# desc_act = True, 0.3191/0.3567
class TestLlama3_2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.3311
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3549
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.05
    APPLY_CHAT_TEMPLATE = True
    V2 = False
    DEBUG = True
    ACT_GROUP_AWARE = False
    DESC_ACT = True
    DATASET_SIZE = 1024
    DATASET_SORT = "desc"
    QUANT_BATCH_SIZE = 4
    # USE_FLASH_ATTN = False
    # EORA = Lora(
    #     # for quant, path is save path. for load, it is loading path
    #     path="./eora_test",
    #     rank=128,
    # )
    # b1 = 0.315, b4 = 0.3106, b8 = 0.3148, b32 = 0.3148, b16 = 0.3234

    def test_llama3_2(self):
        self.quant_lm_eval()
