# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os

import pytest
import torch
from model_test import ModelTest
from transformers import AutoModelForMultimodalLM, DiffusionGemmaConfig

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.diffusion_gemma import DiffusionGemmaQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig


class TestDiffusionGemma(ModelTest):
    """Exercise quantization through the real multimodal generation path."""

    NATIVE_MODEL_ID = "/monster/data/model/diffusiongemma-26B-A4B-it" # google/diffusiongemma-26B-A4B-it
    USE_FLASH_ATTN = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    INFERENCE_PROMPT = "What is the capital city of France?"
    LOAD_BACKEND = BACKEND.AUTO

    def test_diffusion_gemma(self):
        with self.model_compat_test_context():
            model, _, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                need_eval=False,
                call_perform_post_quant_validation=False,
            )

        self.assertIsNotNone(processor)
        image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ovis/10016.jpg",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated = model.generate(**inputs)
        sequences = getattr(generated, "sequences", generated)
        generated_ids = sequences[:, inputs["input_ids"].shape[1] :]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print("output_text", output_text)

        self.assertIn("snow", output_text.lower())


def _tiny_diffusion_gemma_config():
    """The smallest tied DiffusionGemma stack the reload path exercises."""

    return DiffusionGemmaConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 64,
            "num_experts": 4,
            "top_k_experts": 2,
            "max_position_embeddings": 64,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_bidirectional_attention": "vision",
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 64,
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "position_embedding_size": 16,
        },
        canvas_length=4,
        image_token_id=60,
        boi_token_id=61,
        eoi_token_id=62,
    )


@pytest.mark.cpu
def test_quantized_checkpoint_reloads_with_tied_decoder(tmp_path, monkeypatch):
    """Issue #5514: a saved checkpoint must reload with the decoder reattached."""

    torch.manual_seed(0)
    source = tmp_path / "source"
    saved = tmp_path / "quantized"
    source.mkdir()

    model = AutoModelForMultimodalLM.from_config(
        _tiny_diffusion_gemma_config(), dtype=torch.bfloat16
    )
    model.save_pretrained(source)

    # A config-only fixture has no image/audio processor assets; the processor
    # is unrelated to the quantized reload path under test.
    monkeypatch.setattr(DiffusionGemmaQModel, "require_load_processor", False)

    rows = [
        {"input_ids": list(range(1 + i, 49 + i)), "attention_mask": [1] * 48}
        for i in range(8)
    ]
    quantized = GPTQModel.load(
        str(source), GPTQConfig(bits=4, group_size=32), dtype=torch.bfloat16
    )
    quantized.quantize(CalibrationConfig(dataset=rows, batch_size=1))
    quantized.save(str(saved))

    loaded = GPTQModel.load(str(saved), device="cpu")

    encoder_layers = loaded.model.model.encoder.language_model.layers
    decoder_layers = loaded.model.model.decoder.layers
    shared_leaves = 0
    for encoder_layer, decoder_layer in zip(encoder_layers, decoder_layers):
        for name, module in encoder_layer.named_modules():
            if not name or not isinstance(module, BaseQuantLinear):
                continue
            assert decoder_layer.get_submodule(name) is module
            shared_leaves += 1
    assert shared_leaves > 0

    output = loaded.model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    assert output.logits is not None
    assert torch.isfinite(output.logits).all()
