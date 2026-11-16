# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest


class TestSpark2_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Spark-X2.5-4B" # XHToken/Spark-X2.5-4B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 16
    ASSERT_INFERENCE = True
    # Spark answers the generic prompt with the king France is named after,
    # not the city; use the prompt from the #4121 repro, which is deterministic
    # under greedy decoding.
    INFERENCE_PROMPT = "The capital of France is"

    # Full 1172-row ARC-Challenge scores for GPTQ 4-bit (g128), measured against
    # dense BF16 acc=0.36006825938566556 and acc_norm=0.3916382252559727. The
    # quantized checkpoint retains 89.81% / 89.32% of the dense scores.
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.32337883959044367, "floor_pct": 0.04},
            "acc_norm": {"value": 0.34982935153583616, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_spark2_5(self):
        self.quantize_and_evaluate()

    def test_spark2_5_native_generate_runs_before_model_load_hook(self):
        """Issue #4121 item 2: the native branch must call `before_model_load`.

        Spark-X2.5's remote forward passes `input_embeds`/`cache_position` to
        the transformers mask helper; only `Spark2_5QModel.before_model_load`
        adapts those names. Without the hook a native load raises
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
