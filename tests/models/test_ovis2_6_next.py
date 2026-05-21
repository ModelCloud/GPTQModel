# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

import torch
from model_test import ModelTest
from PIL import Image

from gptqmodel.quantization.config import MOE_ALL_EXPERTS, ExpertsRoutingOverride, MoEConfig


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Ovis2.6-80B-A3B"

    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1
    MOE_CONFIG = MoEConfig(ExpertsRoutingOverride(num_experts_per_tok=MOE_ALL_EXPERTS))
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_ovis2_6_moe(self):
        with self.model_compat_test_context():
            model, _tokenizer, _processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        text_tokenizer = model.text_tokenizer

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What does this picture show?"},
            ],
        }]

        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
        )
        input_ids = input_ids.to(model.device)
        pixel_values = pixel_values.to(
            dtype=model.visual_tokenizer.vit.dtype,
            device=model.device,
        ) if pixel_values is not None else None
        grid_thws = grid_thws.to(model.device) if grid_thws is not None else None

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
            )
            output = text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Output:\n{output}")

            self.assertIn("snow", output.lower())
