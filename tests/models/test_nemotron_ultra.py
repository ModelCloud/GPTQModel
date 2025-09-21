# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestNemotronUltra(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3_1-Nemotron-Ultra-253B-v1"
    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True

    def test_nemotron_ultra(self):
        self.quant_lm_eval()
