# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGemma(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-2-9b" # "google/gemma-2-9b"
    NATIVE_ARC_CHALLENGE_ACC = 0.6143
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.6553

    def test_gemma(self):
        self.quant_lm_eval()


