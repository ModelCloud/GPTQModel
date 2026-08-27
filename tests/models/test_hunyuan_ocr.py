# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from gptqmodel import BACKEND
from model_test import ModelTest
from ovis import image_to_test_dataset


def test_prepare_hunyuan_ocr_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("image-url", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_hunyuan_ocr_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_hunyuan_ocr_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image-url"},
                    {
                        "type": "text",
                        "text": (
                            "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
                            "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
                        ),
                    },
                ],
            }
        ]
    ]


class TestHunyuanOCR(ModelTest):
    NATIVE_MODEL_ID = "tencent/HunyuanOCR"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.AUTO
    EVAL_BATCH_SIZE = 16
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.2696245733788396, "floor_pct": 0.04},
            "acc_norm": {"value": 0.30716723549488056, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_hunyuan_ocr(self):
        self.quantize_and_evaluate()
