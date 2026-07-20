# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import test_llama3_2_scale_search as llama_scale_search


class TestQwen3_8BScaleSearchAB(llama_scale_search.TestLlama3_2ScaleSearchAB):
    """Retest projection-scoped scale-search objectives on a larger dense model."""

    # Use the complete local 8B instruction checkpoint and preserve the Llama
    # calibration/evaluation workload so model-size results remain comparable.
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-8B"
    GSM8K_FIXED_BATCH_SIZE = 16
    EVAL_TASKS_FAST = copy.deepcopy(llama_scale_search.TestLlama3_2ScaleSearchAB.EVAL_TASKS_FAST)
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["chat_template"] = False
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_batch_size"] = GSM8K_FIXED_BATCH_SIZE
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["evalution_suite_kwargs"]["batch_size"] = GSM8K_FIXED_BATCH_SIZE
    EVAL_TASKS_FAST["gsm8k_platinum_cot"]["acc,num"]["value"] = 0.5
    EVAL_TASKS_FAST["arc_challenge"]["chat_template"] = False
    EVAL_TASKS_FAST["arc_challenge"]["acc"]["value"] = 0.5
    EVAL_TASKS_FAST["arc_challenge"]["acc_norm"]["value"] = 0.5
    EVAL_TASKS_SLOW = EVAL_TASKS_FAST

    # Avoid collecting the inherited Llama-named full-model A/B. This test is
    # intentionally the ten-arm QKV/O/MLP projection-scope replication.
    test_scale_search_objective_post_quant_ab = None

    def test_qwen3_8b_scale_search_objective_by_projection_scope_post_quant_ab(self):
        """Run the shared QKV/O/MLP objective matrix on Qwen3-8B."""

        super().test_scale_search_objective_by_projection_scope_post_quant_ab()

    test_scale_search_objective_by_projection_scope_post_quant_ab = None
