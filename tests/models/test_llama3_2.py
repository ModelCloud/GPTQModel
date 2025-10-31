# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


# a100:7, MARLIN kernel
# desc_act = False, act_group_aware = False 0.3200/0.3447
# desc_act = False, act_group_aware = True 0.3181/0.3481
# desc_act = True, REGRESSION 0.3191/0.3601
# a100:6+7: MARLIN kernel
# desc_act = False, act_group_aware = True 0.3217/0.3643
# | Metric                         |   MARLIN |
# |--------------------------------|----------|
# | arc_challenge :: acc,none      |   0.3174 |
# | arc_challenge :: acc_norm,none |   0.3601 |
# | mmlu_stem :: acc,none          |   0.3186 |
class TestLlama3_2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3191, # 0.3294 4096, 0.3242 2048
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.3507, # 0.3558 4096, 0.3635 2048
                "floor_pct": 0.04,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.2978, # 0.3099 4096, 0.3270 2048
                "floor_pct": 0.04,
            },
        },
    }

    # llama 3.2 Instruct requires chat = true to have normal ARC scores
    # mmlu requires chat = false
    # APPLY_CHAT_TEMPLATE = True
    # QUANT_BATCH_SIZE = 4

    # EORA = Lora(
    #     # for quant, path is save path. for load, it is loading path
    #     path="./eora_test",
    #     rank=128,
    # )
    # b1 = 0.315, b4 = 0.3106, b8 = 0.3148, b32 = 0.3148, b16 = 0.3234

    def test_llama3_2(self):
        self.quant_lm_eval()
