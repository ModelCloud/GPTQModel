# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from gptqmodel import BACKEND
from model_test import ModelTest
from ovis import image_to_test_dataset


def test_prepare_deepseek_ocr2_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("image-url", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_deepseek_ocr2_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_deepseek_ocr2_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        {
            "image": "image-url",
            "text": "<image>\nFree OCR.",
        }
    ]


class TestDeepSeekOCR2(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/DeepSeek-OCR-2"  # deepseek-community/DeepSeek-OCR-2
    LOAD_BACKEND = BACKEND.AUTO
    USE_FLASH_ATTN = False

    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.3430, "floor_pct": 0.4},
            "acc_norm": {"value": 0.3540, "floor_pct": 0.4},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_deepseek_ocr2(self):
        self.quantize_and_evaluate()
