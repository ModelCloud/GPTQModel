# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in Llama 3.2 1B GPTQ + GSQ checkpoint and GSM8K Platinum test.

ModelTest builds the GPTQ config from this class's settings: 4-bit, group size
128, symmetric weights, desc_act=False, act_group_aware=True, 5% Hessian
damping, and its default RTN fallback at the 0.5% threshold. The config also
uses GPTQ format, true_sequential=True, lm_head=False, and CPU packing. Both
runs use the same calibration and Evalution settings. The GSQ run refines all
seven decoder projections.
GPTQMODEL_LLAMA3_2_GSQ_BASELINE=1 leaves the parent GPTQ config untouched.

With GPTQMODEL_MODEL_TEST_MODE=fast, the harness quantizes the last two decoder
layers by default; GPTQMODEL_FAST_LAYER_COUNT=all covers all 16. Smoke mode
also uses 32 calibration examples, 128 GSM8K questions, and 96 generated
tokens. The ordinary fast run uses 512 calibration examples and the full
GSM8K Platinum task with 256 generated tokens.
"""

import copy
import os
from typing import ClassVar
from unittest.mock import patch

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT, METHOD, GSQConfig
from gptqmodel.quantization import gptq as gptq_mod


class TestLlama3_2GSQ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    SAVE_PATH = os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_SAVE_PATH", "/tmp/llama3_2_gptq_gsq")
    METHOD = METHOD.GPTQ
    FORMAT = FORMAT.GPTQ
    BITS = 4
    GROUP_SIZE = 128
    SYM = True
    DESC_ACT = False
    ACT_GROUP_AWARE = True
    DAMP_PERCENT = 0.05
    DATASET_SIZE = 512
    MODEL_COMPAT_FAST_LAYER_COUNT = 2
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"
    DELETE_QUANTIZED_MODEL = False
    LOAD_BACKEND = BACKEND.TORCH
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST: ClassVar = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 8,
                "max_new_tokens": 256,
                "stream": True,
            },
            # Historical full-task GPTQ expectation, separate from the smoke comparison below.
            "acc,num": {"value": 0.4690, "floor_pct": 0.04, "ceil_pct": 1.0},
        },
    }

    def _build_quantize_config(self):
        config = super()._build_quantize_config()
        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_BASELINE") == "1":
            return config
        config.gsq = GSQConfig(
            enabled=True,
            steps=8,
            candidates=5,
            max_candidate_bytes=32 * 1024**2,
        )
        config.validate_gsq()
        return config

    def test_gsm8k_platinum_gsq(self):
        if os.environ.get("GPTQMODEL_RUN_LLAMA3_2_GSQ_E2E") != "1":
            self.skipTest("Set GPTQMODEL_RUN_LLAMA3_2_GSQ_E2E=1 to run the model quality check")

        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_SMOKE") == "1":
            self.EVAL_TASKS_FAST = copy.deepcopy(self.EVAL_TASKS_FAST)
            task = self.EVAL_TASKS_FAST["gsm8k_platinum_cot"]
            task["evalution_suite_kwargs"]["max_rows"] = 128
            task["evalution_suite_kwargs"]["max_new_tokens"] = 96
            # Matched GPTQ-only smoke baseline: 59/128 on this 32-example calibration.
            task["acc,num"] = {"value": 0.4609375, "floor_pct": 0.08, "ceil_pct": 1.0}
            self.DATASET_SIZE_FAST = 32

        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_BASELINE") == "1":
            self.SAVE_PATH = os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_BASELINE_SAVE_PATH", "/tmp/llama3_2_gptq_gsq_baseline")

        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_EVAL_ONLY") == "1":
            self.get_eval_tasks()
            results = self.evaluate_model(model=self.SAVE_PATH, delete_quantized_model=False)
            self.check_results(results)
            return

        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_BASELINE") == "1":
            with patch.object(gptq_mod, "refine_gptq_scalar", side_effect=AssertionError("GSQ ran in the GPTQ baseline")):
                self.quantize_and_evaluate()
            return

        original = gptq_mod.refine_gptq_scalar
        original_selection = gptq_mod.gsq_enabled_for
        refinements = []
        selected_modules = []

        def counted_selection(config, module_name):
            selected = original_selection(config, module_name)
            if selected:
                selected_modules.append(module_name)
            return selected

        def counted_refinement(*args, **kwargs):
            result = original(*args, **kwargs)
            self.assertLessEqual(result.after, result.before + 1e-7)
            refinements.append((result.before, result.after))
            return result

        with patch.object(gptq_mod, "gsq_enabled_for", counted_selection), patch.object(
            gptq_mod, "refine_gptq_scalar", counted_refinement
        ):
            self.quantize_and_evaluate()
        if self._model_test_mode() == self.MODEL_TEST_MODE_FAST:
            layer_count = self._resolve_fast_model_layer_count_config(16)["resolved"]
            for suffix in (
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ):
                self.assertEqual(
                    sum(name.endswith(suffix) for name in selected_modules), layer_count,
                    f"GSQ did not select every {suffix} projection",
                )
            self.assertEqual(len(refinements), 7 * layer_count, "GSQ did not refine every selected projection")
        else:
            self.assertGreater(len(refinements), 0, "GSQ refinement was not used during model quantization")
