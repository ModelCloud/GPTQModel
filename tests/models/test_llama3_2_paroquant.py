# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.quantization import FORMAT, METHOD, ParoQuantQuantizeConfig


class TestLlama3_2_ParoQuant(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 24,
                "max_new_tokens": 96,
                "streaming": True,
                "max_rows": 128,
            },
            "acc,num": {
                "value": 0.460938,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3216723549488055,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3515358361774744,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.40120520139549637,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.34325889164598844,
                "floor_pct": 0.04,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.30631399317406144,
                "floor_pct": 0.04,
            },
            "acc_norm": {
                "value": 0.33532423208191126,
                "floor_pct": 0.04,
            },
        },
        "mmlu_stem": {
            "chat_template": False,
            "acc": {
                "value": 0.3850301300348874,
                "floor_pct": 0.04,
            },
        },
    }
    FORMAT = FORMAT.PAROQUANT
    METHOD = METHOD.PAROQUANT
    SYM = True
    TORCH_DTYPE = torch.float16
    LOAD_BACKEND = BACKEND.PAROQUANT
    QUANT_BACKEND = BACKEND.PAROQUANT
    KERNEL_QUANT = {ParoQuantQuantLinear}
    KERNEL_INFERENCE = {ParoQuantQuantLinear}

    PAROQUANT_ROTATION_EPOCHS = 5
    PAROQUANT_FINETUNE_EPOCHS = 5
    PAROQUANT_TRAIN_SAMPLES = 512

    def _build_quantize_config(self):
        return ParoQuantQuantizeConfig(
            bits=self.BITS,
            method=METHOD.PAROQUANT,
            format=FORMAT.PAROQUANT,
            opt_rotation_epochs=self.PAROQUANT_ROTATION_EPOCHS,
            opt_finetune_epochs=self.PAROQUANT_FINETUNE_EPOCHS,
            opt_train_samples=self.PAROQUANT_TRAIN_SAMPLES,
            opt_stage_impl="reference",
            opt_pair_impl="fast",
            opt_quantizer_impl="reference",
            adapter=self.EORA,
            vram_strategy=self.VRAM_STRATEGY,
            dynamic=self.DYNAMIC,
            moe=self.MOE_CONFIG,
            offload_to_disk=self.OFFLOAD_TO_DISK,
        )

    def test_llama3_2_paroquant(self):
        self.quant_lm_eval()
