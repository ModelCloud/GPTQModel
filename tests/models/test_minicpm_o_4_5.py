# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os

from model_test import ModelTest
from PIL import Image


class TestMiniCPMO4_5(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/MiniCPM-o-4_5" # openbmb/MiniCPM-o-4_5
    TRUST_REMOTE_CODE = True
    EVAL_BATCH_SIZE = 1

    def test_minicpm_o_4_5(self):
        # Evalution does not support minicpmo, and will throw an error during execution:
        # E TypeError: MiniCPMO.forward() missing 1 required positional argument: 'data
        with self.model_compat_test_context():
            model, tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        image = Image.open(image_path).convert('RGB')

        # First round chat
        question = "What is the landform in the picture?"
        msgs = [{'role': 'user', 'content': [image, question]}]

        answer = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
        )

        generated_text = ""
        for new_text in answer:
            generated_text += new_text

        print(f'Output:\n{generated_text}')

        self.assertIn("snow", generated_text.lower())
