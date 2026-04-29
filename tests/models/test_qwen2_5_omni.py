# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
import unittest
from importlib.metadata import PackageNotFoundError, version

from model_test import ModelTest
from packaging.version import Version

from gptqmodel.models.definitions.qwen2_5_omni import Qwen2_5_OmniGPTQ


class TestQwen2_5_Omni(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-Omni-3B"
    EVAL_TASKS = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.2329, "floor_pct": 0.2},
            "acc_norm": {"value": 0.2765, "floor_pct": 0.2},
        },
    }
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        required = {
            "audioread": Version("3.1.0"),
            "librosa": Version("0.11.0"),
            "av": Version("16.0.1"),
        }
        for pkg, minimum in required.items():
            try:
                installed = Version(version(pkg))
            except PackageNotFoundError:
                raise unittest.SkipTest(
                    f"Qwen2.5 Omni requires {pkg}>={minimum}"
                )

            if installed < minimum:
                raise unittest.SkipTest(
                    f"Qwen2.5 Omni requires {pkg}>={minimum}, found {installed}"
                )

        try:
            version("soundfile")
        except PackageNotFoundError:
            raise unittest.SkipTest("Qwen2.5 Omni requires soundfile")

    def test_qwen2_5_omni(self):
        import soundfile as sf

        model, tokenizer, processor = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                                      dtype=self.TORCH_DTYPE)
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
        generated_ids, audio = self.generate_stable_with_limit(
            model,
            processor,
            inputs=inputs,
            max_new_tokens=128,
            return_generate_output=True,
            return_audio=True,
        )
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
