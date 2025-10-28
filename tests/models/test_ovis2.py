# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

import torch
from model_test import ModelTest
from PIL import Image


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis2-2B-hf"

    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1

    def test_ovis(self):
        model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                           dtype=self.TORCH_DTYPE, batch_size=1)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What does this picture show?"},
                ],
            },
        ]
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path)
        messages = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            images=[image],
            text=messages,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f'Output:\n{output}')

            self.assertIn("snow", output.lower())
