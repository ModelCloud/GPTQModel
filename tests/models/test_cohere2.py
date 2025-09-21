# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestCohere2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/c4ai-command-r7b-12-2024"
    NATIVE_ARC_CHALLENGE_ACC = 0.4680
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4693
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.15
    EVAL_BATCH_SIZE = 4
    USE_FLASH_ATTN = False

    def test_cohere2(self):
        self.quant_lm_eval()
