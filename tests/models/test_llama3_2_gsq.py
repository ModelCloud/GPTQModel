# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in Llama 3.2 1B GPTQ + GSQ checkpoint and GSM8K Platinum test."""

import copy
import os
from unittest.mock import patch

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import GSQConfig, gptq as gptq_mod


class TestLlama3_2GSQ(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    SAVE_PATH = os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_SAVE_PATH", "/tmp/llama3_2_gptq_gsq")
    DELETE_QUANTIZED_MODEL = False
    LOAD_BACKEND = BACKEND.TORCH
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = {
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
            "acc,num": {"value": 0.4690, "floor_pct": 0.04, "ceil_pct": 1.0},
        },
    }

    def _build_quantize_config(self):
        config = super()._build_quantize_config()
        config.gsq = GSQConfig(
            enabled=True,
            steps=8,
            candidates=5,
            max_candidate_bytes=32 * 1024**2,
            modules=(r"self_attn\.q_proj$",),
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
            task["acc,num"] = {"value": 0.4609375, "floor_pct": 0.08, "ceil_pct": 1.0}
            self.DATASET_SIZE_FAST = 32

        if os.environ.get("GPTQMODEL_LLAMA3_2_GSQ_EVAL_ONLY") == "1":
            self.get_eval_tasks()
            results = self.evaluate_model(model=self.SAVE_PATH, delete_quantized_model=False)
            self.check_results(results)
            return

        original = gptq_mod.refine_gptq_scalar
        refinements = []

        def counted_refinement(*args, **kwargs):
            result = original(*args, **kwargs)
            self.assertLessEqual(result.after, result.before + 1e-7)
            refinements.append((result.before, result.after))
            return result

        with patch.object(gptq_mod, "refine_gptq_scalar", counted_refinement):
            self.quantize_and_evaluate()
        self.assertGreater(len(refinements), 0, "GSQ refinement was not used during model quantization")
