# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
from model_test import ModelTest
from PIL import Image
from transformers import AutoModelForImageTextToText

from gptqmodel.models import auto
from gptqmodel.models.definitions.llava import LlavaQModel


def test_llava_model_type_selects_image_text_definition(monkeypatch):
    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="llava"),
    )

    assert auto.check_and_get_model_definition("/tmp/llava") is LlavaQModel
    assert LlavaQModel.loader is AutoModelForImageTextToText


def test_llava_definition_targets_the_nested_llama_decoder():
    modules = LlavaQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flattened = {name for block in modules for name in block}

    assert LlavaQModel.extract_layers_node() == ["model.language_model.layers"]
    assert LlavaQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    } <= flattened


def test_llava_definition_maps_runtime_paths_to_hf_checkpoint_keys():
    rules = LlavaQModel.resolve_hf_conversion_map_reversed()
    mappings = {
        (rule.source_patterns[0], rule.target_patterns[0])
        for rule in rules
    }

    assert (
        r"^model\.language_model\.(.+)$",
        r"^language_model.model.\1",
    ) in mappings
    assert (
        r"^model\.vision_tower\.(.+)$",
        r"^vision_tower.vision_model.\1",
    ) in mappings


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.cuda
class TestLlava15(ModelTest):
    NATIVE_MODEL_ID = os.environ.get("GPTQMODEL_LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 1
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_COUNT = "all"

    def test_llava_15_quantize_and_generate_with_image(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                need_eval=False,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        with Image.open(image_path) as image:
            llava_image = image.convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": llava_image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"LLaVA output:\n{output}")

        self.assertIn("snow", output.lower())
