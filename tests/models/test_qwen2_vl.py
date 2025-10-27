# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from model_test import ModelTest

from gptqmodel.models.definitions.qwen2_vl import Qwen2VLQModel
from gptqmodel.utils.eval import EVAL


class TestQwen2_VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2-VL-2B-Instruct"
    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "chat_template": True,
            "acc": {"value": 0.3524, "floor_pct": 0.2},
            "acc_norm": {"value": 0.3763, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6

    def test_qwen2_vl(self):
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

        image_inputs = Qwen2VLQModel.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print("output_text:", output_text)

        self.assertIn("dog", output_text)


        # check lm_eval results
        self.check_kernel(model, self.KERNEL_INFERENCE)

        task_results = self.lm_eval(model=model,
                                    trust_remote_code=self.TRUST_REMOTE_CODE,
                                    delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)
