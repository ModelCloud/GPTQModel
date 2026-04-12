# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel


class TestQwen3_VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen3-VL-2B-Instruct/"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3618, "floor_pct": 0.04},
            "acc_norm": {"value": 00.3882, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6

    def test_qwen3_vl(self):
        with self.model_compat_test_context():
            model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                                          dtype=self.TORCH_DTYPE)

        # check image to text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = Qwen3_VLQModel.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        output_text = self.generate_stable_with_limit(
            model,
            processor,
            inputs=inputs,
            max_new_tokens=128,
            batch_decode=True,
            clean_up_tokenization_spaces=False,
        )
        print("output_text:", output_text)

        self.assertIn("dog", output_text)


        # check evaluation results
        self.check_kernel(model, self.KERNEL_INFERENCE)

        with self.model_compat_test_context():
            task_results = self.evaluate_model(model=model,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)
