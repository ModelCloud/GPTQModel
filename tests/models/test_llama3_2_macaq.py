# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

import pytest
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.macaq import MacaqTritonLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import MacaqConfig
from gptqmodel.utils.macaq_triton import triton_mxfp4_available


# MACaQ MXFP4 W4A4 on Llama-3.2-1B-Instruct.
#
# Fast-mode baseline (128 calibration chunks, 64 flip sweeps, 128 ARC rows,
# chat template, eager attention, Triton W4A4):
#   arc_challenge :: accuracy,loglikelihood       0.3281
#   arc_challenge :: accuracy,loglikelihood_norm  0.3203
#
# Full-model Wikitext-2 perplexity: 128 Wikitext-2 chunks, 128 flips,
# pre_alpha=0.75, and all 112 projections quantized:
#   BF16 20.50 | MACaQ W4A16 21.85 | MACaQ W4A4 24.22
class TestLlama3_2_MACaQ(ModelTest):
    NATIVE_MODEL_ID = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MODEL",
        "/monster/data/model/Llama-3.2-1B-Instruct",
    )
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_LLAMA3_2_MACAQ_SAVE_PATH",
        "/tmp/llama3_2_macaq_saved_ckpt",
    )
    DELETE_QUANTIZED_MODEL = False
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    DATASET_SIZE_FAST = 64
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False

    METHOD = METHOD.MACAQ
    FORMAT = FORMAT.MACAQ
    BITS = 4
    GROUP_SIZE = 32
    LOAD_BACKEND = BACKEND.MACAQ_TRITON
    QUANT_BACKEND = BACKEND.AUTO
    KERNEL_QUANT = {MacaqTritonLinear}
    KERNEL_INFERENCE = {MacaqTritonLinear}

    MACAQ_N_FLIP_ITERS = 64
    MACAQ_HAD_BLOCK = 32
    MACAQ_PRE_ALPHA = 0.5

    # The stock ModelTest config builder is GPTQ-shaped; MACaQ has its own
    # settings, so build the method config directly.
    def _build_quantize_config(self):
        return MacaqConfig(
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            n_flip_iters=self.MACAQ_N_FLIP_ITERS,
            had_block=self.MACAQ_HAD_BLOCK,
            pre_alpha=self.MACAQ_PRE_ALPHA,
            offload_to_disk=self.OFFLOAD_TO_DISK,
        )

    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "chat_template": True,
            "evalution_suite_kwargs": {"max_rows": 128},
            "acc": {
                "value": 0.328125,
                "floor_pct": 0.10,
                "ceil_pct": 0.30,
            },
            "acc_norm": {
                "value": 0.3203125,
                "floor_pct": 0.10,
                "ceil_pct": 0.30,
            },
        },
    }

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.296875,
                "floor_pct": 0.15,
                "ceil_pct": 0.30,
            },
            "acc_norm": {
                "value": 0.296875,
                "floor_pct": 0.15,
                "ceil_pct": 0.30,
            },
        },
    }

    def test_llama3_2_macaq(self):
        if not triton_mxfp4_available():
            pytest.skip("MACaQ Triton inference requires a CUDA device")
        self.quantize_and_evaluate()
