# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
import types

from model_test import ModelTest
import torch

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
    monkeypatch.setattr(instance, "load_processor", lambda: FakeProcessor())

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
    monkeypatch.setattr(instance, "load_processor", lambda: FakeProcessor())

    instance.prepare_dataset([conversation], batch_size=1)

    assert calls == [[conversation]]


def test_causal_mask_compat_drops_removed_cache_position(monkeypatch):
    module_name = "tests.fake_modeling_intern_s2_preview"
    modeling_module = types.ModuleType(module_name)
    calls = []

    def create_causal_mask(*, config, inputs_embeds, attention_mask, past_key_values):
        calls.append((config, inputs_embeds, attention_mask, past_key_values))
        return "mask"

    modeling_module.create_causal_mask = create_causal_mask
    monkeypatch.setitem(sys.modules, module_name, modeling_module)

    language_model_type = type(
        "FakeInternS2PreviewTextModel",
        (),
        {"__module__": module_name},
    )
    instance = object.__new__(intern_s2_preview.InternS2PreviewQModel)
    instance.model = types.SimpleNamespace(
        model=types.SimpleNamespace(language_model=language_model_type())
    )

    instance.monkey_patch()
    result = modeling_module.create_causal_mask(
        config="config",
        inputs_embeds="inputs",
        attention_mask="attention",
        cache_position="removed argument",
        past_key_values="cache",
    )

    assert result == "mask"
    assert calls == [("config", "inputs", "attention", "cache")]


def test_monkey_patch_uses_remote_stateful_cache(monkeypatch):
    module_name = "tests.fake_modeling_intern_s2_preview_cache"
    modeling_module = types.ModuleType(module_name)

    class InternS2PreviewDynamicCache:
        def get_seq_length(self, layer_idx=0):
            return 5

        def get_mask_sizes(self, cache_position, layer_idx):
            return cache_position.shape[0] + self.get_seq_length(layer_idx), 0

    modeling_module.InternS2PreviewDynamicCache = InternS2PreviewDynamicCache

    def create_causal_mask(*, cache_position=None):
        return cache_position

    modeling_module.create_causal_mask = create_causal_mask
    monkeypatch.setitem(sys.modules, module_name, modeling_module)

    language_model_type = type(
        "FakeInternS2PreviewTextModel",
        (),
        {"__module__": module_name},
    )
    model_type = type(
        "FakeInternS2PreviewForConditionalGeneration",
        (),
        {
            "_is_stateful": True,
            "_supports_default_dynamic_cache": classmethod(lambda cls: True),
        },
    )
    instance = object.__new__(intern_s2_preview.InternS2PreviewQModel)
    instance.model = model_type()
    instance.model.model = types.SimpleNamespace(
        language_model=language_model_type(),
    )

    instance.monkey_patch()

    assert instance.model._supports_default_dynamic_cache() is False
    cache = InternS2PreviewDynamicCache()
    assert cache.get_query_offset(layer_idx=3) == 5
    assert cache.get_mask_sizes(2, layer_idx=3) == (7, 0)
    assert cache.get_mask_sizes(torch.arange(2), layer_idx=3) == (7, 0)


class TestInternS2Preview(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Intern-S2-Preview"
    TRUST_REMOTE_CODE = True
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
