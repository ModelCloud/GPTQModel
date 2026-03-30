# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from model_test import ModelTest

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.quantization import FORMAT, METHOD, ParoQuantizeConfig


def _env_flag(*names: str, default: bool = False) -> bool:
    """Parse the first present boolean env var from a prioritized name list."""
    truthy = {"1", "true", "yes", "on", "y", "t"}
    falsy = {"0", "false", "no", "off", "n", "f"}
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in truthy:
            return True
        if value in falsy:
            return False
    return default


def _env_choice(*names: str, default: str) -> str:
    """Return the first non-empty env override from a prioritized name list."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value:
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    """Return the first parseable integer env override from a prioritized name list."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return int(value)
    return default


class TestLlama3_2_ParoQuant(ModelTest):
    # Keep one stable saved checkpoint so eval-only repro runs can reload the exact post-quant model.
    SAVE_PATH = os.environ.get(
        "GPTQMODEL_PAROQUANT_SUBSECTION_SAVE_PATH",
        os.environ.get(
            "GPTQMODEL_PAROQUANT_SAVE_PATH",
            "/tmp/paroquant_evalution_saved_ckpt_subsection",
        ),
    )
    DELETE_QUANTIZED_MODEL = False
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
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
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
    }
    FORMAT = FORMAT.PAROQUANT
    METHOD = METHOD.PAROQUANT
    SYM = True
    TORCH_DTYPE = torch.bfloat16
    LOAD_BACKEND = BACKEND.PAROQUANT
    QUANT_BACKEND = BACKEND.PAROQUANT
    KERNEL_QUANT = {ParoQuantQuantLinear}
    KERNEL_INFERENCE = {ParoQuantQuantLinear}
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"

    # Accuracy-focused fast-mode defaults: last 4 layers, 2+2 epochs on 4096 train rows.
    PAROQUANT_ROTATION_EPOCHS = 2
    PAROQUANT_FINETUNE_EPOCHS = 2
    PAROQUANT_TRAIN_SAMPLES = 4096
    PAROQUANT_SEED = 3141592653

    @staticmethod
    def _opt_train_on_noisy_inputs() -> bool:
        """Subsection keeps same-stream training by default; noisy-input replay regressed."""
        return _env_flag(
            "GPTQMODEL_PAROQUANT_SUBSECTION_TRAIN_ON_NOISY_INPUTS",
            "GPTQMODEL_PAROQUANT_TRAIN_ON_NOISY_INPUTS",
            default=False,
        )

    @staticmethod
    def _opt_stage_impl() -> str:
        """Allow stage-runner A/Bs without changing the default model test behavior."""
        return _env_choice(
            "GPTQMODEL_PAROQUANT_SUBSECTION_STAGE_IMPL",
            "GPTQMODEL_PAROQUANT_STAGE_IMPL",
            default="fast",
        )

    @classmethod
    def _rotation_epochs(cls) -> int:
        return _env_int(
            "GPTQMODEL_PAROQUANT_SUBSECTION_ROTATION_EPOCHS",
            "GPTQMODEL_PAROQUANT_ROTATION_EPOCHS",
            default=cls.PAROQUANT_ROTATION_EPOCHS,
        )

    @classmethod
    def _finetune_epochs(cls) -> int:
        return _env_int(
            "GPTQMODEL_PAROQUANT_SUBSECTION_FINETUNE_EPOCHS",
            "GPTQMODEL_PAROQUANT_FINETUNE_EPOCHS",
            default=cls.PAROQUANT_FINETUNE_EPOCHS,
        )

    @classmethod
    def _train_samples(cls) -> int:
        return _env_int(
            "GPTQMODEL_PAROQUANT_SUBSECTION_TRAIN_SAMPLES",
            "GPTQMODEL_PAROQUANT_TRAIN_SAMPLES",
            default=cls.PAROQUANT_TRAIN_SAMPLES,
        )

    def _build_quantize_config(self):
        return ParoQuantizeConfig(
            bits=self.BITS,
            method=METHOD.PAROQUANT,
            format=FORMAT.PAROQUANT,
            opt_scope="subsection",
            opt_train_on_noisy_inputs=self._opt_train_on_noisy_inputs(),
            opt_rotation_epochs=self._rotation_epochs(),
            opt_finetune_epochs=self._finetune_epochs(),
            opt_train_samples=self._train_samples(),
            opt_seed=self.PAROQUANT_SEED,
            opt_stage_impl=self._opt_stage_impl(),
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
