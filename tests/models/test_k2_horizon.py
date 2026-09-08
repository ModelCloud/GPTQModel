# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from model_test import ModelTest

from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig


class TestK2Horizon(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/K2-Horizon-0.9B"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 32
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.30802047781569963, "floor_pct": 0.04},
            "acc_norm": {"value": 0.3191126279863481, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_k2_horizon(self):
        self.quantize_and_evaluate()


class TestK2HorizonMoVA(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/K2-Horizon-MoVA-36B-A4B"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 4
    DATASET_SIZE_FAST = 32
    # Full dense BF16 ARC-Challenge baseline (1,172 samples):
    # acc=0.5776450511945392, acc_norm=0.5964163822525598.
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "acc": {"value": 0.5537542662116041, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5776450511945392, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    # K2 has two independently sized routed families (100 MLP experts and 64
    # MoVA value experts). Replay the calibration set through every expert;
    # chunking keeps the per-subset memory bounded.
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingBypass(batch_size=16))
    # The first three layers are dense; exercise a real MoVA/MoE layer in fast mode.
    MODEL_COMPAT_FAST_LAYER_POSITION = "last"

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        # The MoVA checkpoint's chat template requires every historical
        # assistant message to carry an explicit thinking field. The shared
        # calibration corpus does not provide one, but it includes an
        # equivalent pre-rendered text sample. Use that representation rather
        # than inventing reasoning content merely to satisfy the template.
        dataset = super().load_dataset(tokenizer=tokenizer, rows=rows)
        return [{"text": sample["text"]} for sample in dataset]

    def test_k2_horizon_mova(self):
        self.quantize_and_evaluate()
