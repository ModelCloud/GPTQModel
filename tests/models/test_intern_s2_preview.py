# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import sys
import types

from model_test import ModelTest
import torch
from torch import nn

from gptqmodel.models.definitions import intern_s2_preview
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig


def _resolve_model_path() -> str:
    requested_path = "/monster/data/model/internlm/Intern-S2-Preview"
    local_fallback = "/monster/data/model/Intern-S2-Preview"
    if os.path.isdir(requested_path):
        return requested_path
    if os.path.isdir(local_fallback):
        return local_fallback
    return requested_path


def test_capture_modules_follow_quant_device(monkeypatch):
    instance = object.__new__(intern_s2_preview.InternS2PreviewQModel)
    instance.model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                embed_tokens=nn.Embedding(4, 4),
                rotary_emb=nn.Identity(),
            ),
            visual=nn.Identity(),
        )
    )
    quant_device = torch.device("cuda:7")
    instance.quantize_config = types.SimpleNamespace(
        device=quant_device,
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    materialize_devices = []
    instance.shell_module_materialize = lambda module, device: (
        materialize_devices.append(device) or module
    )

    move_devices = []
    monkeypatch.setattr(
        intern_s2_preview,
        "move_to",
        lambda module, device: move_devices.append(device) or module,
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert materialize_devices == [quant_device, quant_device, quant_device]
    assert move_devices == [torch.device("cpu")] * 3


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


class TestInternS2Preview(ModelTest):
    NATIVE_MODEL_ID = _resolve_model_path()
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride())

    def test_intern_s2_preview(self):
        with self.model_compat_test_context():
            model, tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=self.QUANT_BATCH_SIZE,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ovis/10016.jpg",
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What does this picture show?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=self.TORCH_DTYPE)

        output_text = self.generate_stable_with_limit(
            model,
            tokenizer,
            inputs=inputs,
            max_new_tokens=128,
        )
        print("output_text:", output_text)

        self.assertIn("snow", output_text.lower())
