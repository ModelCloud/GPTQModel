# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest
from importlib.metadata import PackageNotFoundError, version

from model_test import ModelTest
from packaging.version import Version


class TestBrumby(ModelTest):
    GROUP_SIZE = 32
    DATASET_SIZE = 1024
    DATASET_SIZE_FAST = 128
    # Brumby decoder layers are structurally uniform, so fast mode can quantize
    # the first layers and avoid replaying 38 untouched layers during calibration.
    MODEL_COMPAT_FAST_LAYER_COUNT = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    OFFLOAD_TO_DISK_FAST = False
    NATIVE_MODEL_ID = "/monster/data/model/Brumby-14B-Base"
    TRUST_REMOTE_CODE = True
    LOAD_MODEL_EXTRA_ARGS = {"use_cache": False}
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.87,
                "floor_pct": 0.05,
                "ceil_pct": 0.10,
            },
        },
        "gsm8k_cot": {
            "chat_template": True,
            "acc,num": {
                "value": 0.88,
                "floor_pct": 0.05,
                "ceil_pct": 0.10,
            },
        },
        "arc_challenge": {
            "acc": {"value": 0.89, "floor_pct": 0.05, "ceil_pct": 0.10},
        },
        "mmlu": {
            "acc": {"value": 0.71, "floor_pct": 0.05, "ceil_pct": 0.10},
        },
    }
    EVAL_TASKS_FAST = {
        "arc_challenge": {
            "evalution_batch_size": 8,
            "evalution_suite_kwargs": {"max_rows": 32},
            "acc": {"value": 0.89, "floor_pct": 0.10, "ceil_pct": 1.0},
        },
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            installed = Version(version("retention"))
        except PackageNotFoundError:
            raise unittest.SkipTest("retention>=1.0.7 is required for Brumby")

        if installed < Version("1.0.7"):
            raise unittest.SkipTest(
                f"retention>=1.0.7 is required for Brumby, found {installed}"
            )

    def test_brumby(self):
        self.quantize_and_evaluate()
