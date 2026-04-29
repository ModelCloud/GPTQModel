# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


# TODO, this model requires 24G vram at least
class TestGptNeoX(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gpt-neox-20b" # "EleutherAI/gpt-neox-20b"
    NATIVE_ARC_CHALLENGE_ACC = 0.3805
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4078
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = NATIVE_ARC_CHALLENGE_ACC_SLOW
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW
    def test_gptneox(self):
        self.quantize_and_evaluate()

