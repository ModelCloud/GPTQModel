# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestLFM3VL(ModelTest):
    # Text decoder only; vision tower and multimodal projector stay unquantized.
    NATIVE_MODEL_ID = "/monster/data/models/LFM3-VL-2B"  # LiquidAI/LFM3-VL-2B
    TRUST_REMOTE_CODE = True
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3114,
                "floor_pct": 0.05,
            },
            "acc_norm": {
                "value": 0.3430,
                "floor_pct": 0.05,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_lfm3_vl(self):
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
