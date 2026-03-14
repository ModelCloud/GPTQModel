# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import builtins
import sys
import types

import pytest
from PIL import Image
from torch import nn

from gptqmodel.models.definitions import base_qwen3_vl


def test_qwen3_vl_video_missing_dependency_has_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "qwen_vl_utils", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_vl_utils.vision_process", raising=False)

    real_import = builtins.__import__

    def fail_qwen_vl_import(name, *args, **kwargs):
        if name.startswith("qwen_vl_utils"):
            raise ImportError("No module named 'qwen_vl_utils'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qwen_vl_import)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "dummy.mp4"},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    with pytest.raises(ImportError, match="pip install qwen-vl-utils"):
        base_qwen3_vl.BaseQwen3VLGPTQ.process_vision_info(messages)


def test_qwen3_vl_image_only_process_vision_info_returns_image_list():
    image = Image.new("RGB", (2, 2), color="white")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    image_inputs = base_qwen3_vl.BaseQwen3VLGPTQ.process_vision_info(messages)

    assert isinstance(image_inputs, list)
    assert image_inputs == [image]


def test_qwen3_vl_pre_quantize_hooks_use_inner_model_layout():
    instance = object.__new__(base_qwen3_vl.BaseQwen3VLGPTQ)
    inner_model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            embed_tokens=nn.Embedding(4, 4),
            rotary_emb=nn.Identity(),
        ),
        visual=nn.Identity(),
    )
    instance.model = types.SimpleNamespace(model=inner_model)
    instance.quantize_config = types.SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert instance.model.model.language_model.embed_tokens.weight.device.type == "cpu"
