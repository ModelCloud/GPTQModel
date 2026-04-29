# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch  # noqa: E402
from model_test import ModelTest


class TestBloom(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/bloom-560m" # "bigscience/bloom-560m"
    NATIVE_ARC_CHALLENGE_ACC = 0.2201
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2440
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = {"A100": 0.2201, "RTX4090": 0.2124}
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = {"A100": 0.2397, "RTX4090": 0.2380}
    TORCH_DTYPE = torch.float16
    USE_FLASH_ATTN = False

    def test_bloom(self):
        self.quantize_and_evaluate()
