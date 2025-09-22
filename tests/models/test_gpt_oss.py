# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestGPTOSS(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-oss-20b-BF16/"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.4411
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4718
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = False
    EVAL_BATCH_SIZE = 6
    USE_VLLM = False

    def test_gpt_oss(self):
        self.quant_lm_eval()
