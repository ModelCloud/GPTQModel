# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestMiniCpm(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-2B-128k"  # "openbmb/MiniCPM-2B-128k"
    NATIVE_ARC_CHALLENGE_ACC = 0.3848
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4164
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 4
    ASSERT_INFERENCE = True

    def test_minicpm(self):
        args = {}

        self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE, **args)


class TestMiniCpmDpo(ModelTest):
    """The package-owned decoder must not read the removed `seen_tokens` field.

    Issue #4121 item 3: transformers 5 removed the attribute, so a native load
    raised `AttributeError: 'DynamicCache' object has no attribute
    'seen_tokens'`. The 128k checkpoint uses a different remote revision, so
    this family needs its own checkpoint to pin the failure.
    """

    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-2B-dpo-bf16"  # "openbmb/MiniCPM-2B-dpo-bf16"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False

    def test_minicpm_dpo_native_generate(self):
        from gptqmodel import GPTQModel

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=None,
            trust_remote_code=False,
            device="cuda:0",
        )
        try:
            self.assertInference(model)
        finally:
            del model
