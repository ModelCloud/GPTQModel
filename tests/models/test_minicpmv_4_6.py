# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestMiniCPMV4_6(ModelTest):
    NATIVE_MODEL_ID = "openbmb/MiniCPM-V-4.6" # openbmb/MiniCPM-V-4.6"
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1

    def test_minicpmv_4_6(self):
        # Evalution does not support minicpmv, and will throw an error during execution:
        # E TypeError: MiniCPMV.forward() missing 1 required positional argument: 'data
        with self.model_compat_test_context():
            model, tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg"),
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=downsample_mode,
            max_slice_nums=36,
        ).to(model.device)

        generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f'Output:\n{output_text}')

        self.assertIn("snow", output_text.lower())
