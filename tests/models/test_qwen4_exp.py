# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest

from gptqmodel.quantization.config import GcMode


class TestQwen3_8FlashNext(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3.8-Flash-Next"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 16
    EVAL_SINGLE_GPU = False

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.6194539249146758, "floor_pct": 0.04},
            "acc_norm": {"value": 0.6100682593856656, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    SAVE_PATH = "./temp/qwen4_exp_test"

    def _build_quantize_config(self):
        config = super()._build_quantize_config()
        # Drain 1,500+ expert pack jobs before replaying the next layer.
        config.wait_for_submodule_finalizers = True
        # Release temporary replay buffers after each stage.
        config.gc_mode = GcMode.ON_STAGE_END
        return config

    def test_qwen3_8_flash_next(self):
        self.quantize_and_evaluate()


__all__ = ["TestQwen3_8FlashNext"]
