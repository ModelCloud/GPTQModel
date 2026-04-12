# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import importlib.util

from model_test import ModelTest


class TestBaiChuan(ModelTest):
    """Compat coverage for Baichuan remote tokenizer loading and monolithic checkpoint handling."""

    NATIVE_MODEL_ID = "/monster/data/model/Baichuan2-7B-Chat" # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4104
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4317
    NATIVE_ARC_CHALLENGE_ACC_SLOW = NATIVE_ARC_CHALLENGE_ACC
    NATIVE_ARC_CHALLENGE_ACC_NORM_SLOW = NATIVE_ARC_CHALLENGE_ACC_NORM
    NATIVE_ARC_CHALLENGE_ACC_FAST = {"A100": 0.3771, "RTX4090": 0.3890}
    NATIVE_ARC_CHALLENGE_ACC_NORM_FAST = {"A100": 0.3890, "RTX4090": 0.4001}
    MODEL_MAX_LEN = 4096
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 6
    OFFLOAD_TO_DISK = False  # Local checkpoint is a monolithic .bin, so LazyTurtle offload is unavailable.

    def test_baichuan(self):
        # Baichuan's remote tokenizer imports sentencepiece eagerly, so skip before model load when absent.
        if importlib.util.find_spec("sentencepiece") is None:
            self.skipTest("Baichuan tokenizer remote code requires sentencepiece")

        self.quantize_and_evaluate()
