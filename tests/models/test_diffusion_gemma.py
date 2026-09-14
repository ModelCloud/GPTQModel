# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from gptqmodel import BACKEND
from model_test import ModelTest
import os

class TestDiffusionGemma(ModelTest):
    """Exercise quantization through the real multimodal generation path."""

    NATIVE_MODEL_ID = "/monster/data/model/diffusiongemma-26B-A4B-it"
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
