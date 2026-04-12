# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestErnie4_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/ERNIE-4.5-0.3B-PT/"
    NATIVE_ARC_CHALLENGE_ACC = 0.25597269624573377
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.30119453924914674
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = 0.25
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = 0.2977815699658703
    EVAL_BATCH_SIZE = 6
    USE_FLASH_ATTN = False

    def test_exaone(self):
        self.quantize_and_evaluate()
