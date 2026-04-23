# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGemma(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-3n-E2B-it" # "google/gemma-3n-E2B-it"
    NATIVE_ARC_CHALLENGE_ACC = 0.4308
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4556
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = 0.37457337883959047
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = 0.3839590443686007

    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_gemma(self):
        self.quantize_and_evaluate()

