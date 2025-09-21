# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGlm4Moe(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/_ci_/GLM-4.5-Air/"
    NATIVE_ARC_CHALLENGE_ACC = 0.3805
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4078
    def test_glm4moe(self):
        self.quant_lm_eval()

