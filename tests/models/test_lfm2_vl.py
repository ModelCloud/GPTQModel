# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestLFM2VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/LFM2.5-VL-1.6B"
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 1

    def test_lfm2_vl(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
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

        output_ids = model.generate(**inputs, max_new_tokens=128)
        output = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"Output:\n{output}")

        self.assertIn("snow", output.lower())
