# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import os.path
from types import SimpleNamespace

import numpy as np
import torch
from model_test import ModelTest
from ovis import image_to_test_dataset
from PIL import Image
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.openvla import OpenVLAQModel
from gptqmodel.models.local import LocalModelLoader


def test_openvla_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="openvla")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "load_model_config", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/openvla-7b") is OpenVLAQModel


def test_openvla_module_tree_covers_llama_decoder_paths():
    layer_modules = OpenVLAQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert OpenVLAQModel.layer_modules_strict is True
    assert not OpenVLAQModel.require_trust_remote_code
    assert isinstance(OpenVLAQModel.loader, LocalModelLoader)
    assert OpenVLAQModel.require_load_processor is True
    assert OpenVLAQModel.support_batch_quantize is False
    assert OpenVLAQModel.lm_head == "language_model.lm_head"
    assert OpenVLAQModel.pre_lm_head_norm_module == "language_model.model.norm"
    assert OpenVLAQModel.rotary_embedding == "language_model.model.rotary_emb"
    assert OpenVLAQModel.extract_layers_node() == ["language_model.model.layers"]

    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules


def test_openvla_base_modules_include_vision_stack_and_language_roots():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(4, 4)
            self.model.layers = nn.ModuleList([nn.Identity()])
            self.model.norm = nn.LayerNorm(4)
            self.model.rotary_emb = nn.Identity()
            self.lm_head = nn.Linear(4, 4, bias=False)

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_backbone = nn.Identity()
            self.projector = nn.Identity()
            self.language_model = _LanguageModel()

    base_modules = set(OpenVLAQModel.get_base_modules(_Wrapper()))

    assert "vision_backbone" in base_modules
    assert "projector" in base_modules
    assert "language_model.model.embed_tokens" in base_modules
    assert "language_model.model.norm" in base_modules
    assert "language_model.model.rotary_emb" in base_modules
    assert "language_model.lm_head" in base_modules
    assert "language_model.model.layers" not in base_modules


def test_openvla_pre_quantize_hook_materializes_base_modules():
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.embed_tokens = nn.Embedding(4, 4)
    model.language_model.model.norm = nn.LayerNorm(4)
    model.language_model.model.rotary_emb = nn.Identity()
    model.vision_backbone = nn.Linear(4, 4)
    model.projector = nn.Linear(4, 4)

    qmodel = object.__new__(OpenVLAQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    materialized = []

    def shell_module_materialize(module, device, module_path=None):
        materialized.append((module, device, module_path))
        return module

    qmodel.shell_module_materialize = shell_module_materialize

    qmodel.pre_quantize_generate_hook_start()

    assert materialized == [
        (model.language_model.model.embed_tokens, torch.device("cpu"), "language_model.model.embed_tokens"),
        (model.language_model.model.norm, torch.device("cpu"), "language_model.model.norm"),
        (model.language_model.model.rotary_emb, torch.device("cpu"), "language_model.model.rotary_emb"),
        (model.vision_backbone, torch.device("cpu"), "vision_backbone"),
        (model.projector, torch.device("cpu"), "projector"),
    ]


def test_openvla_calibration_prompt_uses_reference_template():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8))},
                {"type": "text", "text": "pick up the cup"},
            ],
        },
        {"role": "assistant", "content": "action"},
    ]

    prompt, image = OpenVLAQModel._extract_prompt_and_image(conversation)

    assert prompt == "In: What action should the robot take to pick up the cup?\nOut: "
    assert isinstance(image, Image.Image)

    prompt, _ = OpenVLAQModel._extract_prompt_and_image(
        [{"role": "user", "content": [{"type": "text", "text": "In: wave\nOut:"}]}]
    )
    assert prompt == "In: wave\nOut: "


def test_prepare_openvla_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("https://example.com/cat.jpg", "pick up the cup")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_openvla_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_openvla_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/cat.jpg"},
                    {"type": "text", "text": "pick up the cup"},
                ],
            },
        ]
    ]


def test_get_calib_dataset_dispatches_openvla(monkeypatch):
    calls = {}

    def fake_prepare_openvla_dataset(n_sample):
        calls["n_sample"] = n_sample
        return []

    monkeypatch.setattr(image_to_test_dataset, "prepare_openvla_dataset", fake_prepare_openvla_dataset)

    qmodel = object.__new__(OpenVLAQModel)
    nn.Module.__init__(qmodel)

    assert image_to_test_dataset.get_calib_dataset(qmodel) == []
    assert calls == {"n_sample": 20}


class TestOpenVLA(ModelTest):
    NATIVE_MODEL_ID = "openvla/openvla-7b"
    TRUST_REMOTE_CODE = False

    def test_openvla(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path).convert("RGB")
        prompt = "In: What action should the robot take to pick up the cup?\nOut: "
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print("action:", action)

        assert action.shape == (7,)
        assert np.isfinite(action).all()
        assert (np.abs(action) <= 1.0).all()

        self.check_kernel(model, self.KERNEL_INFERENCE)
