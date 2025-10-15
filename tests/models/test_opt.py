# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# gpu:7, a100
class TestOpt(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/opt-125m"  # "facebook/opt-125m"
    NATIVE_ARC_CHALLENGE_ACC = 0.1920
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2253
    INPUTS_MAX_LENGTH = 2048 # opt embedding is max 2048

    def test_opt(self):
        self.quant_lm_eval()
