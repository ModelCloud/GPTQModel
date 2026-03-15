# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from model_test import ModelTest

from gptqmodel.utils.eval import EVAL


class TestBrumby(ModelTest):
    GROUP_SIZE = 32
    DATASET_SIZE = 1024
    NATIVE_MODEL_ID = "/monster/data/model/Brumby-14B-Base"
    TRUST_REMOTE_CODE = True
    LOAD_MODEL_EXTRA_ARGS = {"use_cache": False}
    DATASET_CONCAT_SIZE = 2048
    EVAL_TASKS_SLOW = {
        EVAL.LM_EVAL.GSM8K_PLATINUM_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.87,
                "floor_pct": 0.05,
                "ceil_pct": 0.10,
            },
        },
        EVAL.LM_EVAL.GSM8K_COT: {
            "chat_template": True,
            "exact_match,flexible-extract": {
                "value": 0.88,
                "floor_pct": 0.05,
                "ceil_pct": 0.10,
            },
        },
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.89, "floor_pct": 0.05, "ceil_pct": 0.10},
        },
        EVAL.LM_EVAL.MMLU: {
            "acc": {"value": 0.71, "floor_pct": 0.05, "ceil_pct": 0.10},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

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
        self.quant_lm_eval()
