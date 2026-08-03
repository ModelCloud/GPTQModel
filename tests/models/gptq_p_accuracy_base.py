# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared base for the per-bit planar (gptq_p) end-to-end post-quant accuracy tests.

Each `tests/models/test_llama3_2_gptq_p_<bits>bit.py` file subclasses this with only
`BITS`, `SAVE_PATH`, and its per-bit expected metrics, so every bit runs as an
independent pytest file (parallelizable one-file-per-GPU).

Config under test: format=gptq_p, GAR (act_group_aware=True, which forces
desc_act=False), scale_search=activation, group_size=128 (GAR auto-disables at
group_size<=32), TritonV2 planar load path.
"""

import os

from model_test import ModelTest

from gptqmodel import BACKEND, ScaleSearchConfig
from gptqmodel.quantization import FORMAT


class GptqPAccuracyBase(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    FORMAT = FORMAT.GPTQ_P
    DESC_ACT = False
    ACT_GROUP_AWARE = True  # GAR; ModelTest forces desc_act=False when GAR is on
    SCALE_SEARCH = ScaleSearchConfig.ACTIVATION
    GROUP_SIZE = 128
    # Marlin has no planar support; TritonV2 serves gptq_p bits 2-8 (Pangolin
    # native GEMV takes over decode shapes at post_init where eligible).
    LOAD_BACKEND = BACKEND.TRITON
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    # Keep the per-bit snapshots on disk so checkpoint sizes can be collected after a run.
    DELETE_QUANTIZED_MODEL = False

    def _model_test_mode(self) -> str:
        # The gptq_p per-bit matrix is a full regression suite: default to slow
        # (full-model) quantization unless the mode env is explicitly set.
        if not os.environ.get(self.MODEL_TEST_MODE_ENV, "").strip():
            return self.MODEL_TEST_MODE_SLOW
        return super()._model_test_mode()

    @classmethod
    def arc_challenge_tasks(cls, acc, acc_norm, floor_pct=0.05, gsm8k=None, gsm8k_floor_pct=0.06):
        """ARC-Challenge plus optional GSM8K-Platinum task table for the per-bit matrix."""
        tasks = {
            "arc_challenge": {
                "chat_template": True,
                "acc": {
                    "value": acc,
                    "floor_pct": floor_pct,
                    "ceil_pct": 1.0,
                },
                "acc_norm": {
                    "value": acc_norm,
                    "floor_pct": floor_pct,
                    "ceil_pct": 1.0,
                },
            },
        }
        if gsm8k is not None:
            tasks["gsm8k_platinum_cot"] = {
                "chat_template": True,
                "evalution_use_model_path": True,
                "evalution_batch_size": "auto",
                "evalution_model_args": {
                    "dtype": "bfloat16",
                    "attn_implementation": "paged|flash_attention_2",
                    "device": "cuda:0",
                },
                "evalution_suite_kwargs": {
                    "batch_size": 32,
                    "max_new_tokens": 256,
                    "stream": True,
                },
                "acc,num": {
                    "value": gsm8k,
                    "floor_pct": gsm8k_floor_pct,
                    "ceil_pct": 1.0,
                },
            }
        return tasks
