# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestTinyllama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
    NATIVE_ARC_CHALLENGE_ACC = 0.2995
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3268
    TRUST_REMOTE_CODE = True

    def test_tinyllama(self):
        self.quant_lm_eval()
