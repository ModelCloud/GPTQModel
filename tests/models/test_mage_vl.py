# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path
from types import SimpleNamespace

import pytest
from model_test import ModelTest
from ovis import image_to_test_dataset
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText

from gptqmodel.models import auto
from gptqmodel.models.definitions import mage_vl
from gptqmodel.models.definitions.mage_vl import MageVLQModel


MODEL_PATH = Path("/monster/data/model/Mage-VL")


def test_mage_vl_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="mage_vl")

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert (
        auto.check_and_get_model_definition("/tmp/mage-vl", trust_remote_code=True)
        is MageVLQModel
    )


def test_mage_vl_definition_matches_qwen3_decoder_contract():
    layer_modules = MageVLQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    assert MageVLQModel.extract_layers_node() == [
        "model.language_model.layers",
        "language_model.layers",
    ]
    assert MageVLQModel.loader is AutoModelForImageTextToText
    assert MageVLQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert MageVLQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert MageVLQModel.out_of_model_tensors == {
        "files": [
            "streammind_gate.safetensors",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
        ],
    }
    assert MageVLQModel.require_load_processor is True
    assert MageVLQModel.require_trust_remote_code is True


def test_mage_vl_processor_load_enables_remote_code(monkeypatch):
    calls = {}
    expected_processor = object()

    def fake_from_pretrained(model_path, **kwargs):
        calls["model_path"] = model_path
        calls.update(kwargs)
        return expected_processor

    monkeypatch.setattr(mage_vl.AutoProcessor, "from_pretrained", fake_from_pretrained)

    qmodel = object.__new__(MageVLQModel)
    qmodel.model_local_path = "/tmp/mage-vl"

    assert qmodel.load_processor() is expected_processor
    assert calls == {
        "model_path": "/tmp/mage-vl",
        "trust_remote_code": True,
    }


def test_mage_vl_uses_qwen_style_image_calibration_dataset(monkeypatch):
    calls = {}
    expected_dataset = object()

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return expected_dataset

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    qmodel = object.__new__(MageVLQModel)

    assert image_to_test_dataset.get_calib_dataset(qmodel) is expected_dataset
    assert calls == {
        "format_func": image_to_test_dataset.format_qwen2_vl_dataset,
        "n_sample": 20,
    }


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Mage-VL model not found")
def test_mage_vl_processor_prepares_image_calibration_sample():
    qmodel = object.__new__(MageVLQModel)
    qmodel.model_local_path = str(MODEL_PATH)
    image = Image.open(MODEL_PATH / "examples/dog.jpg").convert("RGB")
    calibration_dataset = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": "A dog is running on grass."},
        ],
    ]

    [inputs] = qmodel.prepare_dataset(calibration_dataset, batch_size=1)

    image_token_count = int((inputs["input_ids"] == 151655).sum().item())
    merged_patch_count = int((inputs["image_grid_thw"].prod(dim=-1) // 4).sum().item())
    assert image_token_count == merged_patch_count
    assert inputs["pixel_values"].shape[0] == int(
        inputs["image_grid_thw"].prod(dim=-1).sum().item()
    )
    assert inputs["patch_positions"].shape == (inputs["pixel_values"].shape[0], 3)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Mage-VL model not found")
def test_mage_vl_native_shell_matches_definition_tree():
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with init_empty_weights(include_buffers=True):
        shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

    layer = shell.model.language_model.layers[0]

    assert config.model_type == "mage_vl"
    assert config.text_config.model_type == "qwen3"
    assert (
        auto.check_and_get_model_definition(MODEL_PATH, trust_remote_code=True)
        is MageVLQModel
    )
    assert MageVLQModel.get_base_modules(shell) == ["model.visual"]
    assert hasattr(shell.model, "visual")
    assert hasattr(shell.model, "language_model")
    assert hasattr(layer.self_attn, "q_proj")
    assert hasattr(layer.self_attn, "k_proj")
    assert hasattr(layer.self_attn, "v_proj")
    assert hasattr(layer.self_attn, "o_proj")
    assert hasattr(layer.self_attn, "q_norm")
    assert hasattr(layer.self_attn, "k_norm")
    assert hasattr(layer.mlp, "gate_proj")
    assert hasattr(layer.mlp, "up_proj")
    assert hasattr(layer.mlp, "down_proj")


class TestMageVL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Mage-VL"
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False
    EVAL_BATCH_SIZE = 1

    def test_mage_vl(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        image = Image.open(MODEL_PATH / "examples/dog.jpg").convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What animal is shown in this image?"},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output = processor.decode(
            output_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print("output:", output)

        self.assertIn("dog", output.lower())
        self.check_kernel(model, self.KERNEL_INFERENCE)
