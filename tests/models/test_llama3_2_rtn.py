# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import WeightOnlyConfig, SmoothMAD
from gptqmodel.utils.eval import EVAL


class TestLlama3_2_rtn(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048

    FORMAT = FORMAT.GPTQ
    WEIGHT_ONLY = WeightOnlyConfig(smooth=SmoothMAD())
    ACT_GROUP_AWARE = False
    LOAD_BACKEND = BACKEND.TORCH
    USE_FLASH_ATTN = False

    EVAL_TASKS = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.3143,
                "floor_pct": 0.15,
                "ceil_pct": 0.15,
            },
        },
        EVAL.LM_EVAL.MMLU_STEM: {
            "chat_template": False,
            "acc": {
                "value": 0.3990,
                "floor_pct": 0.15,
                "ceil_pct": 0.15,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {
                "value": 0.3089,
                "floor_pct": 0.15,
                "ceil_pct": 0.15,
            },
            "acc_norm": {
                "value": 0.3481,
                "floor_pct": 0.15,
                "ceil_pct": 0.15,
            },
        },
    }

    def test_llama3_2_rtn(self):
        model, tokenizer = self.quantModel(
            self.NATIVE_MODEL_ID,
            batch_size=self.QUANT_BATCH_SIZE,
            trust_remote_code=self.TRUST_REMOTE_CODE,
            dtype=self.TORCH_DTYPE,
            call_perform_post_quant_validation=False,
        )
        try:
            prompt = self.INFERENCE_PROMPT
            output = self.generate_with_limit(model, tokenizer, prompt, max_new_tokens=32)
            self.assertGreater(len(output), len(prompt))
        finally:
            self._cleanup_quantized_model(model, enabled=self.DELETE_QUANTIZED_MODEL)
