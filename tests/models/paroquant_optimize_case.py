# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from model_test import ModelTest, _env_choice, _env_flag, _env_int, _env_optional_flag

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.quantization import FORMAT, METHOD, ParoConfig


def _resolve_save_path(scope_env: str, default: str) -> str:
    """Resolve a scope-specific saved checkpoint path with global fallback."""
    return os.environ.get(
        scope_env,
        os.environ.get("GPTQMODEL_PAROQUANT_SAVE_PATH", default),
    )


PAROQUANT_EVAL_TASKS_FAST = {
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
    # "mmlu_stem": {
    #     "chat_template": False,
    #     "acc": {
    #         "value": 0.40120520139549637,
    #         "floor_pct": 0.04,
    #         "ceil_pct": 1.0,
    #     },
    # },
}

PAROQUANT_EVAL_TASKS_SLOW = {
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
    # "mmlu_stem": {
    #     "chat_template": False,
    #     "acc": {
    #         "value": 0.3850301300348874,
    #         "floor_pct": 0.04,
    #     },
    # },
}


class BaseLlama3_2ParoQuantOptimizeTest(ModelTest):
    """Shared accuracy-oriented ParoQuant optimize test configuration."""

    __test__ = False

    DELETE_QUANTIZED_MODEL = False
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    EVAL_BATCH_SIZE = 64
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_FAST = PAROQUANT_EVAL_TASKS_FAST
    EVAL_TASKS_SLOW = PAROQUANT_EVAL_TASKS_SLOW
    FORMAT = FORMAT.PAROQUANT
    METHOD = METHOD.PARO
    SYM = True
    TORCH_DTYPE = torch.bfloat16
    LOAD_BACKEND = BACKEND.PARO
    QUANT_BACKEND = BACKEND.PARO
    KERNEL_QUANT = {ParoLinear}
    KERNEL_INFERENCE = {ParoLinear}
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"

    # Accuracy-focused fast-mode defaults: last 4 layers, 2+2 epochs on 4096 train rows.
    PAROQUANT_ROTATION_EPOCHS = 2
    PAROQUANT_FINETUNE_EPOCHS = 2
    PAROQUANT_TRAIN_SAMPLES = 4096
    PAROQUANT_SEED = 3141592653

    OPT_SCOPE: str = ""
    TRAIN_ON_NOISY_INPUTS_DEFAULT = False

    @classmethod
    def _scope_prefix(cls) -> str:
        if not cls.OPT_SCOPE:
            raise ValueError(f"{cls.__name__} must define OPT_SCOPE.")
        return cls.OPT_SCOPE.upper()

    @classmethod
    def _opt_train_on_noisy_inputs(cls) -> bool:
        if cls.OPT_SCOPE not in {"layer", "compute_block"}:
            return False
        prefix = cls._scope_prefix()
        return _env_flag(
            f"GPTQMODEL_PAROQUANT_{prefix}_TRAIN_ON_NOISY_INPUTS",
            "GPTQMODEL_PAROQUANT_TRAIN_ON_NOISY_INPUTS",
            default=cls.TRAIN_ON_NOISY_INPUTS_DEFAULT,
        )

    @classmethod
    def _opt_stage_impl(cls) -> str:
        """Allow stage-runner A/Bs without changing the default model test behavior."""
        prefix = cls._scope_prefix()
        return _env_choice(
            f"GPTQMODEL_PAROQUANT_{prefix}_STAGE_IMPL",
            "GPTQMODEL_PAROQUANT_STAGE_IMPL",
            default="fast",
        )

    @classmethod
    def _opt_gradient_checkpointing(cls):
        """Allow scoped activation-checkpointing overrides while keeping config defaults meaningful."""
        prefix = cls._scope_prefix()
        return _env_optional_flag(
            f"GPTQMODEL_PAROQUANT_{prefix}_GRADIENT_CHECKPOINTING",
            "GPTQMODEL_PAROQUANT_GRADIENT_CHECKPOINTING",
        )

    @classmethod
    def _rotation_epochs(cls) -> int:
        prefix = cls._scope_prefix()
        return _env_int(
            f"GPTQMODEL_PAROQUANT_{prefix}_ROTATION_EPOCHS",
            "GPTQMODEL_PAROQUANT_ROTATION_EPOCHS",
            default=cls.PAROQUANT_ROTATION_EPOCHS,
        )

    @classmethod
    def _finetune_epochs(cls) -> int:
        prefix = cls._scope_prefix()
        return _env_int(
            f"GPTQMODEL_PAROQUANT_{prefix}_FINETUNE_EPOCHS",
            "GPTQMODEL_PAROQUANT_FINETUNE_EPOCHS",
            default=cls.PAROQUANT_FINETUNE_EPOCHS,
        )

    @classmethod
    def _train_samples(cls) -> int:
        prefix = cls._scope_prefix()
        return _env_int(
            f"GPTQMODEL_PAROQUANT_{prefix}_TRAIN_SAMPLES",
            "GPTQMODEL_PAROQUANT_TRAIN_SAMPLES",
            default=cls.PAROQUANT_TRAIN_SAMPLES,
        )

    def _build_quantize_config(self):
        return ParoConfig(
            bits=self.BITS,
            method=METHOD.PARO,
            format=FORMAT.PAROQUANT,
            opt_scope=self.OPT_SCOPE,
            opt_train_on_noisy_inputs=self._opt_train_on_noisy_inputs(),
            opt_gradient_checkpointing=self._opt_gradient_checkpointing(),
            opt_rotation_epochs=self._rotation_epochs(),
            opt_finetune_epochs=self._finetune_epochs(),
            opt_train_samples=self._train_samples(),
            opt_seed=self.PAROQUANT_SEED,
            opt_stage_impl=self._opt_stage_impl(),
            opt_pair_impl="fast",
            opt_quantizer_impl="reference",
            adapter=self.EORA,
            dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
            dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
            moe_vram_strategy=self.MOE_VRAM_STRATEGY,
            moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
            dynamic=self.DYNAMIC,
            moe=self.MOE_CONFIG,
            offload_to_disk=self.OFFLOAD_TO_DISK,
        )

    def test_llama3_2_paroquant(self):
        self.quantize_and_evaluate()
