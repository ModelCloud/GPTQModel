# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

import torch
from model_test import ModelTest
from PIL import Image


class TestOvis1_6_Llama(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis1.6-Llama3.2-3B"

    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1
    USE_FLASH_ATTN = False

    def test_ovis_1_6(self):
        # the evaluation harness does not support Ovis, and will throw an error during execution:
        # TypeError: Ovis.forward() missing 3 required positional arguments: 'attention_mask', 'labels', and 'pixel_values'
        model, tokenizer, _ = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                              dtype=self.TORCH_DTYPE, multimodal_max_length=8192, batch_size=1, call_perform_post_quant_validation=False)

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        # enter image path and prompt
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path)
        text = "What does this picture show?"
        query = f'<image>\n{text}'

        # format conversation
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }

        # generate output
        with torch.inference_mode():
            output = self.generate_stable_with_limit(
                model,
                text_tokenizer,
                inputs=inputs,
                max_new_tokens=1024,
                skip_special_tokens=True,
                use_cache=True,
            )

            print(f'Output:\n{output}')

            self.assertIn("snow", output.lower())
