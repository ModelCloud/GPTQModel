# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestSmolVLM(ModelTest):
    NATIVE_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct" # HuggingFaceTB/SmolVLM2-2.2B-Instruct
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.4061, "floor_pct": 0.04},
            "acc_norm": {"value": 0.4394, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    SAVE_PATH = "./temp/smolvlm"

    def test_smolvlm(self):
        self.quantize_and_evaluate()
        model = self.model
        processor = model.processor

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
