# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest


class TestOuro(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ouro-1.4B" # ByteDance/Ouro-1.4B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 32
    ASSERT_INFERENCE = True

    # Full 1172-row ARC-Challenge scores for GPTQ 4-bit (g128), measured against
    # dense BF16 acc=0.5017064846416383 and acc_norm=0.5264505119453925. This is
    # 89.46% / 92.22% dense recovery after Ouro reuses every quantized layer for
    # all four recurrent steps.
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.44880546075085326, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4854948805460751, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_ouro(self):
        self.quantize_and_evaluate()

    def test_ouro_native_generate_runs_before_model_load_hook(self):
        """Issue #4121 item 2: the native branch must call `before_model_load`.

        Ouro's remote forward passes `input_embeds`/`cache_position` to the
        transformers mask helper; only `OuroQModel.before_model_load` adapts
        those names. Without the hook a native load raises
        `TypeError: create_causal_mask() got an unexpected keyword argument
        'input_embeds'`.
        """

        from gptqmodel import GPTQModel

        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=None,
            trust_remote_code=True,
            device="cuda:0",
        )
        try:
            self.assertInference(model)
        finally:
            del model
