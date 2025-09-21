# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestInternlm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/internlm-7b" # "internlm/internlm-7b"
    NATIVE_ARC_CHALLENGE_ACC = 0.4164
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4309
    TRUST_REMOTE_CODE = True
    USE_VLLM = False
    USE_FLASH_ATTN = False

    def test_internlm(self):
        # transformers<=4.44.2 run normal
        self.quant_lm_eval()
