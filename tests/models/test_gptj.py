# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402
from model_test import ModelTest


class TestGptJ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-j-6b" # "EleutherAI/gpt-j-6b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3396
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3660
    TORCH_DTYPE = torch.float16
    INPUTS_MAX_LENGTH = 1024
    USE_FLASH_ATTN = False

    def test_gptj(self):
        self.quant_lm_eval()

