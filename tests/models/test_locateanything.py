# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest
from ovis import image_to_test_dataset


def test_prepare_locateanything_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("image-url", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_locateanything_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_locateanything_dataset,
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
                        "text": "Describe this image and transcribe any visible text.",
                    },
                ],
            },
            {"role": "assistant", "content": "caption"},
        ]
    ]


class TestLocateAnything(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/LocateAnything-3B"  # nvidia/LocateAnything-3B
    TRUST_REMOTE_CODE = True
    # The bundled Qwen forward implements its mask path for SDPA (and Magi),
    # but rejects both eager and FlashAttention 2. The compatibility shim maps
    # ModelTest's eager request to SDPA for this checkpoint.
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 16
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.2099, "floor_pct": 0.04},
            "acc_norm": {"value": 0.2688, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_locateanything(self):
        self.quantize_and_evaluate()
