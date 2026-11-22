# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path

import torch
from model_test import ModelTest
from ovis import image_to_test_dataset

from gptqmodel import BACKEND


def test_prepare_unlimited_ocr_dataset_reuses_shared_dataset(monkeypatch):
    calls = {}

    def fake_prepare_dataset(format_func, n_sample):
        calls["format_func"] = format_func
        calls["n_sample"] = n_sample
        return [format_func("image-url", "caption")]

    monkeypatch.setattr(image_to_test_dataset, "prepare_dataset", fake_prepare_dataset)

    dataset = image_to_test_dataset.prepare_unlimited_ocr_dataset(n_sample=3)

    assert calls == {
        "format_func": image_to_test_dataset.format_unlimited_ocr_dataset,
        "n_sample": 3,
    }
    assert dataset == [
        {
            "image": "image-url",
            "text": "<image>\nFree OCR.",
        }
    ]


class TestUnlimitedOCR(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Unlimited-OCR"  # baidu/Unlimited-OCR
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    LOAD_PROCESSOR = True
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    LOAD_BACKEND = BACKEND.AUTO

    def test_unlimited_ocr(self):
        with self.model_compat_test_context():
            self.model, _tokenizer, _processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                need_eval=False,
                call_perform_post_quant_validation=False,
            )

        self.check_kernel(self.model, self.KERNEL_INFERENCE)
        self._assert_real_image_ocr()

    def _assert_real_image_ocr(self):
        image_path = Path(__file__).resolve().parent / "ovis" / "baidu.png"
        self.assertTrue(image_path.is_file())

        inputs = self.model.prepare_dataset(
            [{"image": str(image_path), "text": "<image>\nFree OCR."}]
        )[0]
        data_device = self.model.model.model.language_model.embed_tokens.weight.device
        inputs = self.model.move_input_capture_example(inputs, data_device)

        prompt_length = inputs["input_ids"].shape[1]
        tokenizer = getattr(self.model.tokenizer, "tokenizer", self.model.tokenizer)

        with torch.inference_mode(), torch.autocast(
            device_type=data_device.type,
            dtype=torch.bfloat16,
            enabled=data_device.type == "cuda",
        ):
            generated_ids = self.model.model.generate(
                **inputs, max_new_tokens=24, do_sample=False, use_cache=True,
            )

        output = tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=True
        )
        print("Unlimited-OCR image output:", output)
        self.assertIn("baidu", output.lower())
