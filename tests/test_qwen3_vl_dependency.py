# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import builtins
import sys
import types

import pytest
import torch
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


def test_qwen3_vl_pre_quantize_hooks_support_direct_layout():
    instance = object.__new__(base_qwen3_vl.BaseQwen3VLGPTQ)
    instance.model = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            embed_tokens=nn.Embedding(4, 4),
            rotary_emb=nn.Identity(),
        ),
        visual=nn.Identity(),
    )
    instance.quantize_config = types.SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    instance.pre_quantize_generate_hook_start()
    instance.pre_quantize_generate_hook_end()

    assert instance.model.language_model.embed_tokens.weight.device.type == "cpu"


def test_qwen3_vl_layout_resolution_supports_nested_wrapper():
    class _InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([nn.Identity()])
            self.visual = nn.Identity()
            self.vision_router = nn.Identity()

    class _OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _InnerModel()

    model = _OuterModel()

    assert base_qwen3_vl.BaseQwen3VLGPTQ.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]
    assert base_qwen3_vl.BaseQwen3VLGPTQ.get_base_modules(model) == ["model.visual", "model.vision_router"]


def test_qwen3_vl_pre_quantize_hooks_materialize_meta_modules_with_nested_layout():
    instance = object.__new__(base_qwen3_vl.BaseQwen3VLGPTQ)
    instance.model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                embed_tokens=nn.Embedding(4, 4, device="meta"),
                rotary_emb=nn.Linear(4, 4, device="meta"),
            ),
            visual=nn.Linear(4, 4, device="meta"),
        )
    )
    instance.quantize_config = types.SimpleNamespace(
        device="cpu",
        offload_to_disk=False,
        offload_to_disk_path="/tmp/unused",
    )

    materialized = {}

    def fake_materialize(module, device):
        replacement = nn.Linear(4, 4) if isinstance(module, nn.Linear) else nn.Embedding(4, 4)
        materialized[id(module)] = (replacement, device)
        return replacement

    instance.shell_module_materialize = fake_materialize

    instance.pre_quantize_generate_hook_start()

    assert instance.model.model.visual.weight.device == torch.device("cpu")
    assert instance.model.model.language_model.embed_tokens.weight.device == torch.device("cpu")
    assert instance.model.model.language_model.rotary_emb.weight.device == torch.device("cpu")
    assert len(materialized) == 3
