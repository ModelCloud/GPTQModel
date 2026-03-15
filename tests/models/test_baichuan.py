# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestBaiChuan(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Baichuan2-7B-Chat" # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4104
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4317
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = 0.3993174061433447
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = 0.42235494880546076
    MODEL_MAX_LEN = 4096
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 6

    def test_baichuan(self):
        self.quant_lm_eval()
