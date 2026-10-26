# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestGemma(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/gemma-3-4b-it" # "google/gemma-3-4b-it"
    # Multimodal Gemma 3 checkpoints calibrate on the shared image-caption
    # dataset, so the text-benchmark baselines are recorded from that
    # image-calibrated path (see test_smolvlm.py for the same VL setup).
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": False,
            "acc": {"value": 0.4846, "floor_pct": 0.04},
            "acc_norm": {"value": 0.5026, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_gemma(self):
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

        output_ids = model.generate(**inputs, max_new_tokens=64)
        output = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"Output:\n{output}")

        self.assertIn("snow", output.lower())
