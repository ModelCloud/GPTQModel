# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from model_test import ModelTest

from gptqmodel.models.definitions import intern_s2_preview


def test_prepare_dataset_wraps_raw_text_as_chat_messages(monkeypatch):
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, conversations, **kwargs):
            calls.append((conversations, kwargs))
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    instance = object.__new__(intern_s2_preview.InternS2PreviewQModel)
    instance.__dict__["processor"] = FakeProcessor()

    prepared = instance.prepare_dataset(["first", "second"], batch_size=2)

    assert len(prepared) == 1
    assert calls == [
        (
            [
                [{"role": "user", "content": [{"type": "text", "text": "first"}]}],
                [{"role": "user", "content": [{"type": "text", "text": "second"}]}],
            ],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "processor_kwargs": {"padding": True},
                "return_dict": True,
                "return_tensors": "pt",
            },
        )
    ]


def test_prepare_dataset_preserves_multimodal_conversations(monkeypatch):
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, conversations, **kwargs):
            calls.append(conversations)
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.jpg"},
                {"type": "text", "text": "describe it"},
            ],
        }
    ]
    instance = object.__new__(intern_s2_preview.InternS2PreviewQModel)
    instance.__dict__["processor"] = FakeProcessor()

    instance.prepare_dataset([conversation], batch_size=1)

    assert calls == [[conversation]]


class TestInternS2Preview(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Intern-S2-Preview"
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    TORCH_DTYPE = torch.bfloat16
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.4283, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4070, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    EVAL_BATCH_SIZE = 18
    EVAL_SINGLE_GPU = False

    def test_intern_s2_preview(self):
        self.quantize_and_evaluate()
