# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestPhi_4(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Phi-4-multimodal-instruct" # "microsoft/Phi-3-mini-4k-instruct"
    NATIVE_ARC_CHALLENGE_ACC = 0.5401
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.5674
    APPLY_CHAT_TEMPLATE = True
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 1

    def test_phi_4(self):
        self.quant_lm_eval()
