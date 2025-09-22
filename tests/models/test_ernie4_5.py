# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestErnie4_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/ERNIE-4.5-0.3B-PT/"
    NATIVE_ARC_CHALLENGE_ACC = 0.2969
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3183
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    USE_FLASH_ATTN = False

    def test_exaone(self):
        self.quant_lm_eval()


