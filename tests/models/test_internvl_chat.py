# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path

from model_test import ModelTest
from PIL import Image


class TestInternvlChat(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/InternVL3_5-1B-Instruct/" # OpenGVLab/InternVL3_5-1B-Instruct
    # InternVL-Chat does not support text-only forwarding.
    # TypeError: InternVLChatModel.forward() missing 1 required positional argument: 'pixel_values'
    # EVAL_TASKS_SLOW = {
    #     "arc_challenge": {
    #         "chat_template": True,
    #         "acc": {"value": 0.3618, "floor_pct": 0.04},
    #         "acc_norm": {"value": 00.3882, "floor_pct": 0.04},
    #     },
    # }
    # EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 6
    OFFLOAD_TO_DISK = False
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"


    def test_internvl_chat(self):
        with self.model_compat_test_context():
            model, tokenizer, _processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    image,
                    {"type": "text", "text": "What does this picture show?"},
                ],
            }
        ]

        inputs = model.prepare_inputs_for_conversation(messages)
        inputs = model.move_input_capture_example(inputs, model.device)
        model.model.img_context_token_id = inputs.pop("img_context_token_id")
        inputs.pop("eos_token_id")

        output_text = self.generate_stable_with_limit(
            model,
            tokenizer,
            inputs=inputs,
            max_new_tokens=128,
        )
        print("output_text:", output_text)

        self.assertIn("snow", output_text.lower())
