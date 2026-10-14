# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import numpy as np
from model_test import ModelTest


def _synthetic_audio(sample_rate: int = 16000) -> np.ndarray:
    """Deterministic tone used as a dependency-free audio input for the smoke run."""

    timeline = np.arange(sample_rate // 2, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * timeline / sample_rate).astype(np.float32)


class TestQwen2Audio(ModelTest):
    NATIVE_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
    TRUST_REMOTE_CODE = False
    EVAL_BATCH_SIZE = 1

    def test_qwen2_audio(self):
        # lm-eval cannot drive Qwen2-Audio's audio-aware chat template, so quantize and
        # validate multimodal generation on the reloaded checkpoint instead.
        with self.model_compat_test_context():
            model, _tokenizer, _processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                call_perform_post_quant_validation=False,
            )

        processor = model.processor
        audio = _synthetic_audio()

        text_messages = [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]
        text = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        output = self.generate_stable_with_limit(
            model,
            processor,
            inputs=inputs,
            max_new_tokens=32,
            batch_decode=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"Text output:\n{output}")
        self.assertTrue(output.strip())

        audio_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "What do you hear?"},
                ],
            }
        ]
        # The audio-aware chat template lives in the processor; a text-only tokenizer
        # template would fail here after reload.
        text = processor.apply_chat_template(audio_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, audio=[audio], return_tensors="pt").to(model.device)
        self.assertIn("input_features", inputs)

        output = self.generate_stable_with_limit(
            model,
            processor,
            inputs=inputs,
            max_new_tokens=32,
            batch_decode=True,
            clean_up_tokenization_spaces=False,
        )
        print(f"Audio output:\n{output}")
        self.assertTrue(output.strip())

        self.check_kernel(model, self.KERNEL_INFERENCE)
