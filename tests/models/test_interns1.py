# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest


class TestInternS1(ModelTest):
    NATIVE_MODEL_ID = "./tmp/Intern-S1-mini"
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_interns1(self):
        with self.model_compat_test_context():
            model, tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image_path},
                    {"type": "text", "text": "What does this picture show?"},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
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
