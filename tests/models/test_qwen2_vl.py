# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gptqmodel.models.definitions.qwen2_vl import Qwen2VLGPTQ
from model_test import ModelTest


class TestQwen2_VL(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2-VL-2B-Instruct"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2329
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2765
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    BATCH_SIZE = 6

    def test_qwen2_vl(self):
        model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                           torch_dtype=self.TORCH_DTYPE)

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

        image_inputs = Qwen2VLGPTQ.process_vision_info(messages)
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
                                    apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                    trust_remote_code=self.TRUST_REMOTE_CODE,
                                    delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)
