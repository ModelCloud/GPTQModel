# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import unittest  # noqa: E402
import tempfile
from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from parameterized import parameterized  # noqa: E402
from datasets import load_dataset
import json
from tokenicer.const import VALIDATE_JSON_FILE_NAME, VALIDATE_ENCODE_PARAMS
from tokenicer.config import ValidateConfig


class TestTokenicer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
        self.model = GPTQModel.load(self.pretrained_model_id, quantize_config=QuantizeConfig(bits=4, group_size=128))
        self.tokenizer = self.model.tokenizer
        self.example = 'Test Case String'
        self.expect_input_ids = [2271, 11538, 923]

    def test_tokenicer_func(self):
        input_ids = self.tokenizer(self.example)['input_ids']
        self.assertEqual(
            input_ids,
            self.expect_input_ids,
            msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
        )

    @parameterized.expand(
        [
            ('eos_token', "<|im_end|>"),
            ('pad_token', "<|fim_pad|>"),
            ('vocab_size', 151643)
        ]
    )
    def test_tokenicer_property(self, property, expect_token):
        if property == 'eos_token':
            result = self.tokenizer.eos_token
        elif property == 'pad_token':
            result = self.tokenizer.pad_token
        elif property == 'vocab_size':
            result = self.tokenizer.vocab_size

        self.assertEqual(
            result,
            expect_token,
            msg=f"Expected property result='{expect_token}' but got '{result}'."
        )

    def test_tokenicer_encode(self):
         input_ids = self.tokenizer.encode(self.example, add_special_tokens=False)
         self.assertEqual(
             input_ids,
             self.expect_input_ids,
             msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
         )

    def test_tokenicer_decode(self):
        example = self.tokenizer.decode(self.expect_input_ids, skip_special_tokens=True)
        self.assertEqual(
            self.example,
            example,
            msg=f"Expected example='{self.example}' but got '{example}'."
        )

    def test_tokenicer_save(self):
        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz",
                                 split="train")
        calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(32))]

        self.model.quantize(calibration_dataset, batch_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model.save(tmpdir)
            validate_json_path = os.path.join(tmpdir, VALIDATE_JSON_FILE_NAME)

            result = os.path.isfile(validate_json_path)
            self.assertTrue(result, f"Save verify file failed: {validate_json_path} does not exist.")

            with open(validate_json_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())

            config = ValidateConfig.from_dict(data)

            validate = True
            for data in config.data:
                input = data.input
                tokenized = self.tokenizer.encode_plus(input, **VALIDATE_ENCODE_PARAMS)["input_ids"].tolist()[0]
                if data.output != tokenized:
                    validate = False
                    break

            self.assertTrue(validate, f"Expected validate='True' but got '{validate}'.")

