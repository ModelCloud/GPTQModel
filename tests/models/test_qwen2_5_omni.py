# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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
import os

import soundfile as sf
from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ
from model_test import ModelTest


class TestQwen2_5_Omni(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-Omni-3B"
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    NATIVE_ARC_CHALLENGE_ACC = 0.2329
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2765
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = True
    EVAL_BATCH_SIZE = 6

    def test_qwen2_5_omni(self):
        model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                           torch_dtype=self.TORCH_DTYPE)
        spk_path = self.NATIVE_MODEL_ID + '/spk_dict.pt'
        model.model.load_speakers(spk_path)

        # check image to text
        messages = [
                {
                    "role": "system",
                    "content": [
                        {   "type": "text",
                            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
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

        image_inputs = Qwen2_5_OmniGPTQ.process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output (text and audio)
        audio_file_name = 'output_gptq.wav'
        generated_ids, audio = model.generate(**inputs, max_new_tokens=128, return_audio = True)
        sf.write(
            audio_file_name,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print("output_text:", output_text)

        self.assertIn("dog", output_text)
        self.assertTrue(os.path.exists(audio_file_name))

        self.check_kernel(model, self.KERNEL_INFERENCE)

        # delete audio file
        os.remove(audio_file_name)

