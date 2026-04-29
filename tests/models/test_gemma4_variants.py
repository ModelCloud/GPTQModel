# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest

from huggingface_hub import snapshot_download
from model_test import ModelTest

from gptqmodel.quantization.config import GcMode, VramStrategy


def _ensure_local_model_dir(local_path: str, repo_id: str) -> str:
    """Download the checkpoint into the shared local model cache when it is missing."""

    if os.path.isdir(local_path):
        return local_path

    os.makedirs(local_path, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_path


class _Gemma4VariantModelTest(ModelTest):
    """Shared Gemma 4 model-test harness tuned for fast variant coverage."""

    # Allow the harness to refresh expectations from the current native model when these baselines drift.
    DISABLE_NATIVE_BASELINE_FALLBACK = False
    TRUST_REMOTE_CODE = False
    TORCH_DTYPE = "bfloat16"
    # Gemma 4 full-attention layers expand to 512-dim heads, which FlashAttention cannot execute.
    USE_FLASH_ATTN = False
    # Gemma 4 variants differ most at the tail: KV sharing, full-attention-only layers, and per-layer adapters.
    MODEL_COMPAT_FAST_LAYER_COUNT = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"
    DATASET_SIZE = 128
    DATASET_CONCAT_SIZE = 1024
    EVAL_BATCH_SIZE = 4
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.30, "floor_pct": 0.35, "ceil_pct": 1.0},
            "acc_norm": {"value": 0.33, "floor_pct": 0.35, "ceil_pct": 1.0},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    HF_MODEL_ID = None

    @classmethod
    def setUpClass(cls):
        if isinstance(getattr(cls, "NATIVE_MODEL_ID", None), str):
            model_path = cls.NATIVE_MODEL_ID.strip()
            if os.path.isabs(model_path) and not os.path.isdir(model_path):
                if not cls.HF_MODEL_ID:
                    raise unittest.SkipTest(f"Model path missing and no HF repo configured: {model_path}")
                cls.NATIVE_MODEL_ID = _ensure_local_model_dir(model_path, cls.HF_MODEL_ID)
        super().setUpClass()


class TestGemma4E2B(_Gemma4VariantModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-4-E2B"
    HF_MODEL_ID = "google/gemma-4-e2b-it"
    EVAL_BATCH_SIZE = 8

    def test_gemma4_e2b(self):
        self.quantize_and_evaluate()


class TestGemma4E4BIt(_Gemma4VariantModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-4-E4B-it"
    HF_MODEL_ID = "google/gemma-4-e4b-it"
    EVAL_BATCH_SIZE = 4

    def test_gemma4_e4b_it(self):
        self.quantize_and_evaluate()


class TestGemma431BIt(_Gemma4VariantModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-4-31B-it"
    HF_MODEL_ID = "google/gemma-4-31b-it"
    EVAL_BATCH_SIZE = 1
    DENSE_VRAM_STRATEGY = VramStrategy.BALANCED

    def _build_quantize_config(self):
        quantize_config = super()._build_quantize_config()
        # 31B full-attention q_proj hits a very large Hessian inverse; flush prior finalizers before the next stage.
        quantize_config.wait_for_submodule_finalizers = True
        quantize_config.gc_mode = GcMode.ON_STAGE_END
        return quantize_config

    def test_gemma4_31b_it(self):
        self.quantize_and_evaluate()
